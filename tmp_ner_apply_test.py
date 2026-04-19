import pandas as pd
from src.nlp.ner import apply_ner_to_df

sample = pd.DataFrame({'text_clean': [
    'ابن عمي نمر نافز التلولي الو خمس ايام مفقود راح علي دوار الكويت علي المساعدات وانفقد',
    'السلام عليكم اخوي الاسير محمود علي بهادر عن حاجز جحر الديك بدنا اي معلومه عنه'
]})

result = apply_ner_to_df(sample, text_col='text_clean')
print(result[['text_clean', 'ner_extracted', 'names', 'location', 'dates']])
