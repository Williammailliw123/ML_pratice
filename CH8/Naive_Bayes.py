import pandas as pd
import numpy as np

def process_email(text):
    text=text.lower()
    return list(set(text.split()))

def count_words(emails):
    model={}
    for index,email in emails.iterrows():
        for word in email['words']:
            if word not in model:
                model[word]={'spam':1,"ham":1}
            else:
                if email['spam']:
                    model[word]['spam']+=1
                else:
                    model[word]['ham']+=1
    return model

def predict_naive_bayes(target_email,emails,model):
    total=len(emails)
    num_spam=sum(emails['spam'])
    num_ham=total-num_spam
    target_email=target_email.lower()
    words=list(set(target_email.split()))
    spam=np.array([])
    ham=np.array([])
    for word in words:
        if word in model:
            spam=np.append(spam,model[word]['spam']/num_spam*total)
            ham=np.append(ham,model[word]['ham']/num_ham*total)
    return np.longlong(np.prod(spam)*num_spam)/(np.longlong(np.prod(spam)*num_spam)+np.longlong(np.prod(ham)*num_ham))

def main():
    emails=pd.read_csv('./CH8/emails.csv')
    emails['words']=emails['text'].apply(process_email)
    #emails.drop('text',axis=1)
    model=count_words(emails)
    target_email=input("Input the email content:")
    result=predict_naive_bayes(target_email,emails,model)
    print(f"The spam ratio is {result}")

if __name__ == "__main__":
    main()