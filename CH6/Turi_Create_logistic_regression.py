import turicreate as tc

def check_three_words(model):
    weights = model.coefficients
    print(f"{weights[weights['index']=='wonderful']}")
    print(f"{weights[weights['index']=='horrible']}")
    print(f"{weights[weights['index']=='the']}")

def main():
    movies=tc.SFrame('IMDB_Dataset.csv')
    movies['words']=tc.text_analytics.count_words(movies['review'])
    model=tc.logistic_classifier.create(movies,features=['words'],target='sentiment')
    check_three_words(model)
    movies['predictions']=model.predict(movies,output_type='probability')
    #print(f"{movies.sort('predictions')[-1]}")
    #print(f"{movies.sort('predictions')[0]}")
    
if __name__ == '__main__':
    main()
    
