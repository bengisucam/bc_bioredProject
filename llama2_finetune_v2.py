
from functools import partial
import os
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, BitsAndBytesConfig, \
    DataCollatorForLanguageModeling, Trainer, TrainingArguments

os.environ["TRANSFORMERS_CACHE"] = "./hf_cache"

model_name = "/data/projects/llm_workspace/llama_models/Llama-2-13b-hf"
dataset_path = "/data/projects/llm_workspace/dataset/large_dataset"
per_device_train_batch_size = 1
gradient_accumulation_steps = 4
warmup_steps = 2
max_steps = 10
learning_rate = 2e-4
fp16 = True
logging_steps = 1
output_dir = "./outputs"
optim = "paged_adamw_8bit"


def load_model(model_name, bnb_config):
    n_gpus = torch.cuda.device_count()
    max_memory = f'{40960}MB'

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",  # dispatch efficiently the model on the available ressources
        max_memory={i: max_memory for i in range(n_gpus)},
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def get_max_length(model):
    conf = model.config
    max_length = None
    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(model.config, length_setting, None)
        if max_length:
            print(f"Found max lenth: {max_length}")
            break
    if not max_length:
        max_length = 1024
        print(f"Using default max length: {max_length}")
    return max_length


def preprocess_batch(batch, tokenizer, max_length):
    """
    Tokenizing a batch
    """
    return tokenizer(
        batch["text"],
        max_length=max_length,
        truncation=True,
    )


def preprocess_dataset(tokenizer: AutoTokenizer, max_length: int, seed, dataset: str):
    """Format & tokenize it so it is ready for training
    :param tokenizer (AutoTokenizer): Model Tokenizer
    :param max_length (int): Maximum number of tokens to emit from tokenizer
    """

    # Add prompt to each sample
    print("Preprocessing dataset...")

    _preprocessing_function = partial(preprocess_batch, max_length=max_length, tokenizer=tokenizer)
    dataset = dataset.map(
        _preprocessing_function,
        batched=True,
        remove_columns=["text"],
    )

    # Filter out samples that have input_ids exceeding max_length
    dataset = dataset.filter(lambda sample: len(sample["input_ids"]) < max_length)

    # Shuffle dataset
    dataset = dataset.shuffle(seed=seed)

    return dataset


def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit  # if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def print_trainable_parameters(model, use_4bit=False):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    if use_4bit:
        trainable_params /= 2
    print(
        f"all params: {all_param:,d} || trainable params: {trainable_params:,d} || trainable%: {100 * trainable_params / all_param}"
    )


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model, tokenizer = load_model(model_name, bnb_config)

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
modules = find_all_linear_names(model)

peft_config = LoraConfig(
    r=2,  # dimension of the updated matrices
    lora_alpha=64,  # parameter for scaling
    target_modules=modules,
    lora_dropout=0.1,  # dropout probability for layers
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, peft_config)
print_trainable_parameters(model)

max_length = get_max_length(model)
dataset = prepare_dataset(dataset_path)
dataset = preprocess_dataset(tokenizer, max_length, 0, dataset)

logger = task.get_logger()
iteration_num = 0


def train():
    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        args=TrainingArguments(
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps, warmup_steps=warmup_steps,
            max_steps=max_steps,
            learning_rate=learning_rate,
            fp16=fp16,
            logging_steps=logging_steps,
            report_to=report_to,
            output_dir=output_dir,
            optim=optim,
        ),
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    model.config.use_cache = False
    trainer.train()

    trainer.save_model("./ft-llama2")


train()
task.close()