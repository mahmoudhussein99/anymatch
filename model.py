import torch
from transformers import T5Tokenizer, GPT2Tokenizer, BertTokenizer, AutoModelForSeq2SeqLM, \
    GPT2ForSequenceClassification, BertForSequenceClassification, AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig

from peft import LoraConfig, get_peft_model
class Matcher:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.print_model_info()

    def print_model_info(self):
        print(f"trainable params: {self.model.num_parameters()}", flush=True)


class T5Matcher(Matcher):
    def __init__(self, base_model: str = 't5-base'):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
        self.tokenizer = T5Tokenizer.from_pretrained(base_model)
        super().__init__(self.model, self.tokenizer)


class GPTMatcher(Matcher):
    def __init__(self, base_model: str = 'gpt2'):
        self.model = GPT2ForSequenceClassification.from_pretrained(base_model)
        self.model.config.pad_token_id = self.model.config.eos_token_id
        self.tokenizer = GPT2Tokenizer.from_pretrained(base_model)
        super().__init__(self.model, self.tokenizer)


class LlamaMatcher(Matcher):
    def __init__(
        self, 
        base_model: str = "meta-llama/Llama-3-8B-Instruct", 
        lora_r: int = 8,  # Reduced LoRA rank for less memory
        lora_alpha: int = 16,  # Lower scaling factor
        lora_dropout: float = 0.1,  # Slightly higher dropout for regularization
        quantization: bool = True,  # Enable quantization
        gradient_checkpointing: bool = True  # Enable gradient checkpointing
    ):
        """
        Initializes a memory-optimized LlamaMatcher with LoRA-Q, quantization, and gradient checkpointing.

        Args:
            base_model (str): Model name or path.
            lora_r (int): LoRA rank (lower values use less memory).
            lora_alpha (int): Scaling factor for LoRA.
            lora_dropout (float): Dropout rate for LoRA.
            quantization (bool): Enable 4-bit quantization using bitsandbytes.
            gradient_checkpointing (bool): Enable gradient checkpointing for reduced memory usage.
        """

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.tokenizer.pad_token = self.tokenizer.eos_token  # Ensure correct padding

        # Configure quantization
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,  # Use 4-bit quantization
            bnb_4bit_compute_dtype=torch.float16,  # Keeps computation in FP16
            bnb_4bit_use_double_quant=True,  # Double quantization for further memory reduction
            bnb_4bit_quant_type="nf4",  # NF4 (Normal Float 4) is the best for Llama
        ) if quantization else None

        # Load base model with quantization (if enabled)
        model = AutoModelForSequenceClassification.from_pretrained(
            base_model, 
            quantization_config=quant_config if quantization else None,
            torch_dtype=torch.float16, 
            device_map="auto"  # Automatically distribute model across available devices
        )

        # Enable gradient checkpointing (saves memory by recomputing activations)
        if gradient_checkpointing:
            model.gradient_checkpointing_enable()

        # Apply LoRA
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="SEQ_CLS"
        )
        self.model = get_peft_model(model, lora_config)
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        super().__init__(self.model, self.tokenizer)

        # Print trainable parameters (useful for debugging)
        self.model.print_trainable_parameters()

class BertMatcher(Matcher):
    def __init__(self, base_model: str = 'bert-base-uncased'):
        self.model = BertForSequenceClassification.from_pretrained(base_model)
        self.tokenizer = BertTokenizer.from_pretrained(base_model)
        super().__init__(self.model, self.tokenizer)


def load_model(base_model):
    if 't5' in base_model:
        model = T5Matcher(base_model)
    elif 'gpt' in base_model:
        model = GPTMatcher(base_model)
    elif 'Llama' in base_model:
        model = LlamaMatcher(base_model)
    elif 'bert' in base_model:
        model = BertMatcher('bert-base-uncased')
    else:
        raise ValueError('Model not found.')
    return model.model, model.tokenizer
