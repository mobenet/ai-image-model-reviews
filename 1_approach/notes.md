FIRST APPROACH: 
- take the 22k dataset from scrapping reddit
- filter the reviews with less than 512 tokens using autotokenizer from cardiffnlp -> 19k
- apply sentiment analysis to the 19k reviews 
- keep just 5k reviews where there is a balanced number of Models + sentiment_label to have a proportion between neg, pos and neutral
