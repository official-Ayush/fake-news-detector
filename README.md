# Fake News Detector 📰

a machine learning project i built to detect whether a news article is real or fake. 
it uses NLP and logistic regression trained on around 40,000+ news articles.

live demo 👉 [click here](https://your-username-fake-news-detector.streamlit.app)

---

## what it does

you paste any news headline or article into the app and it tells you if its fake or real, 
along with a confidence score. pretty useful considering how much misinformation is out there lol

---

## how i built it

honestly took me a while to figure out all the pieces but here's the stack i ended up using:

- **python** - main language
- **scikit-learn** - for the logistic regression model
- **TF-IDF vectorizer** - converts text into numbers the model can understand
- **NLTK** - text cleaning and stopword removal
- **streamlit** - for the web interface
- **pandas** - data loading and preprocessing

---

## model performance

got around 98% accuracy on the test set which i was pretty happy with
```
              precision    recall  f1-score
Fake News       0.98        0.99      0.98
Real News       0.99        0.98      0.98
accuracy                              0.98
```

---

## how to run it locally

first clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/fake-news-detector.git
cd fake-news-detector
```

install the dependencies
```bash
pip install -r requirements.txt
```

download the dataset from kaggle (Fake.csv and True.csv) and put them in the project folder
then train the model
```bash
python train_model.py
```

run the app
```bash
streamlit run app.py
```

thats it, should open at http://localhost:8501

---

## dataset

used the [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) from kaggle.
it has around 23,000 fake articles and 21,000 real articles.

---

## project structure
```
fake-news-detector/
├── app.py              # streamlit web app
├── train_model.py      # model training script
├── model.pkl           # saved trained model
├── vectorizer.pkl      # saved TF-IDF vectorizer
├── requirements.txt    
└── README.md           
```

---

## things i want to add later

- support for URLs (paste a link and it checks the article automatically)
- show which words contributed most to the prediction
- maybe try a more advanced model like BERT

---

## what i learned

this was my first proper end to end ML project. the hardest part was honestly the text 
cleaning and figuring out why the model was overfitting at first. also deploying on 
streamlit was way easier than i expected

if you have any suggestions or find any bugs feel free to open an issue!

---

made by [YOUR NAME](https://github.com/YOUR_USERNAME)
