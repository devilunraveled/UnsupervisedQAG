
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("potsawee/t5-large-generation-race-QuestionAnswer")
model = AutoModelForSeq2SeqLM.from_pretrained("potsawee/t5-large-generation-race-QuestionAnswer")

context = r"""
World number one Novak Djokovic says he is hoping for a "positive decision" to allow him 
to play at Indian Wells and the Miami Open next month. The United States has extended 
its requirement for international visitors to be vaccinated against Covid-19. Proof of vaccination 
will be required to enter the country until at least 10 April, but the Serbian has previously 
said he is unvaccinated. The 35-year-old has applied for special permission to enter the country. 
Indian Wells and the Miami Open - two of the most prestigious tournaments on the tennis calendar 
outside the Grand Slams - start on 6 and 20 March respectively. Djokovic says he will return to 
the ATP tour in Dubai next week after claiming a record-extending 10th Australian Open title 
and a record-equalling 22nd Grand Slam men's title last month.""".replace("\n", "")

inputs = tokenizer(context, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
question_answer = tokenizer.decode(outputs[0], skip_special_tokens=False)
question_answer = question_answer.replace(tokenizer.pad_token, "").replace(tokenizer.eos_token, "")
question, answer = question_answer.split(tokenizer.sep_token)

print("---"*20)
print(f"Output : {question_answer}")
print("---"*20)
print("question:", question)
print("answer:", answer)
