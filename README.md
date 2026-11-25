# Arabic book search and Q&A

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-154F5B?style=flat)

Most NLP tools are built for English first. For this NLP course project, we took Arabic books from the web and ran the whole pipeline ourselves: scraping, cleaning, spelling correction, stemming, then a category classifier and a simple question-answering system.

This was a two-person project with [@Noureldien334](https://github.com/Noureldien334), and I wrote about half of the code. We kept everything in [his repo](https://github.com/Noureldien334/Arabic-NLP), so the commit history shows his uploads. This repo is my fork of it.

## What we did

**Phase 1: build and clean the dataset**

1. **Scraped** 200 pages of [arab-books.com](https://www.arab-books.com/) with BeautifulSoup. We collected 5,992 books with name, author, category, description, publisher, page count and file size.
2. **Cleaned** out English words, numbers and HTML tags, and dropped books with missing names, descriptions or categories. 2,792 books were left.
3. **Corrected spelling** in book names and categories with Hunspell and an Arabic dictionary.
4. **Stemmed** the descriptions with Farasa. We also tried Qalsadi, Snowball, ISRI and MADAMIRA, and Farasa worked best on our text.

**Phase 2: use it**

5. **Predicted a book's category from its description.** We used TF-IDF features (1,000 words) and a random forest, with 70% of the books for training and 30% for testing.
6. **Answered questions.** We built two question-and-answer datasets about the books. A user's question is matched to the stored question that shares the most words with it, and the answer to that question is returned.
7. **Found keywords.** We wrote TF-IDF by hand to weight the words in every description and list the 50 most distinctive ones.

## Results

| Task | Result |
|---|---|
| Category prediction (test set) | **73% accuracy** |
| Category prediction (training set) | 99% accuracy |

There are 70 categories, and the biggest one covers 24% of the books. Always guessing that category would get about 24%, so 73% is well above chance.

## Caveats

- **The classifier overfits.** 99% on training data against 73% on test data is a big gap. Fewer trees, a larger minimum leaf size, or a simpler model like logistic regression would probably generalise better.
- **Accuracy hides a lot with 70 uneven classes.** Macro F1 and a confusion matrix would show which categories the model actually gets right.
- **The TF-IDF vocabulary was built from the raw descriptions** but applied to the stemmed ones, so some stemmed words never match a feature. Fitting it on the stemmed text would be more consistent.
- **The question answering is word overlap.** It doesn't understand synonyms or word order, and Arabic's rich morphology makes exact word matches rarer than in English.
- **The classification notebook reads `books_dataset.csv`,** which isn't in this repo. The cleaned data in `Csv Files/` is the closest match.
- **The name says "recommendation",** but nothing in the code recommends books yet.

## What's in the repo

| Path | What it is |
|---|---|
| `NLP(Phase1).ipynb` | Scraping and cleaning |
| `NLP_Phase_1&2.ipynb` | Everything: scraping, cleaning, classification, Q&A and keywords |
| `Csv Files/` | The scraped data, before and after cleaning |
| `Questions DA/` | The two Q&A datasets |
| `Arabic Dictionary/` | Hunspell dictionary files for spelling correction |

## How to run

The notebooks were written for Google Colab.

1. Open `NLP_Phase_1&2.ipynb` in Colab.
2. Upload the files the cells ask for: the CSVs from `Csv Files/` and `Questions DA/`, and `ar.aff` and `ar.dic` from `Arabic Dictionary/`.
3. Run the cells in order. They install `farasapy` and `hunspell` themselves.

The scraping cells hit the live website, so they may give different results today, or none if the site has changed. You can skip them and start from the saved CSVs.

## What I'd like to do next

Swap TF-IDF for an Arabic BERT model such as MARBERT for both classification and question matching, and actually build the recommender: suggest books with similar descriptions.
