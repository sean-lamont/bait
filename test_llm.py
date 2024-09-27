
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

# quant_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.float16,
#     bnb_4bit_use_double_quant=False
# )

# quant_config = BitsAndBytesConfig(
#     load_in_8bit=True,
#     llm_int8_threshold=6.0
# )

# tokenizer = AutoTokenizer.from_pretrained("EleutherAI/llemma_7b")
# model = AutoModelForCausalLM.from_pretrained("EleutherAI/llemma_7b", quantization_config=quant_config, device_map={"": 0})


# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", quantization_config=quant_config,
#                                              device_map={"": 0})


tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/deepseek-math-7b-base')
model = AutoModelForCausalLM.from_pretrained('deepseek-ai/deepseek-math-7b-base', device_map='auto')


output = model.generate(tokenizer.encode(['Hello']).input_ids)
print (tokenizer.decode(output))
