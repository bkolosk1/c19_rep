
# COVID19 representations


## INSTALLATION

The package can be installed via pip: 

```
pip3 install c19_rep
```
And that's it. You can also install the library directly:

```
python3 setup.py install
```

Note that some of the representations use nltk addons.

```
import nltk
nltk.download('punct')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
```

## USAGE


```python

from c19 import representations
from c19.representations.factorization import SVD
from c19.representations.statistical import Stat
from c19.representations.sent_trans import BERTTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


def run():
    X_train = ["The CDC currently reports 99031 deaths. In general the discrepancies in death counts between different sources are small and explicable. The death toll stands at roughly 100000 people today.", "States reported 1121 deaths a small rise from last Tuesday. Southern states reported 640 of those deaths. https://t.co/YASGRTT4ux","Politically Correct Woman (Almost) Uses Pandemic as Excuse Not to Reuse Plastic Bag https://t.co/thF8GuNFPe #coronavirus #nashville","#IndiaFightsCorona: We have 1524 #COVID testing laboratories in India and as on 25th August 2020 36827520 tests have been done : @ProfBhargava DG @ICMRDELHI #StaySafe #IndiaWillWin https://t.co/Yh3ZxknnhZ", "Populous states can generate large case counts but if you look at the new cases per million today 9 smaller states are showing more cases per million than California or Texas: AL AR ID KS KY LA MS NV and SC. https://t.co/1pYW6cWRaS","Covid Act Now found on average each person in Illinois with COVID-19 is infecting 1.11 other people. Data shows that the infection growth rate has declined over time this factors in the stay-at-home order and other restrictions put in place. https://t.co/hhigDd24fE", "If you tested positive for #COVID19 and have no symptoms stay home and away from other people. Learn more about CDC’s recommendations about when you can be around others after COVID-19 infection: https://t.co/z5kkXpqkYb. https://t.co/9PaMy0Rxaf","Obama Calls Trump’s Coronavirus Response A Chaotic Disaster https://t.co/DeDqZEhAsB","???Clearly, the Obama administration did not leave any kind of game plan for something like this.??�","Retraction—Hydroxychloroquine or chloroquine with or without a macrolide for treatment of COVID-19: a multinational registry analysis - The Lancet https://t.co/L5V2x6G9or" ]
    y_train = [1,1,0,1,1,1,1,0,0,0]
    X_test = ["Take simple daily precautions to help prevent the spread of respiratory illnesses like #COVID19. Learn how to protect yourself from coronavirus (COVID-19): https://t.co/uArGZTrH5L. https://t.co/biZTxtUKyK","The NBA is poised to restart this month. In March we reported on how the Utah Jazz got 58 coronavirus tests in a matter of hours at a time when U.S. testing was sluggish. https://t.co/I8YjjrNoTh https://t.co/o0Nk6gpyos","We just announced that the first participants in each age cohort have been dosed in the Phase 2 study of our mRNA vaccine (mRNA-1273) against novel coronavirus. Read more: https://t.co/woPlKz1bZC #mRNA https://t.co/9VGUoJu5cS"]
    y_test = [1,0,1]
    
    for (name,representation) in [("statistical",Stat), ("SVD",SVD), ("sentence_transformers",BERTTransformer)]:
        representation = representation()
        train_representation = representation.fit_transform(X_train)
        test_representation = representation.transform(X_test)
        clf = LogisticRegression(random_state=0).fit(train_representation, y_train)
        test_predict = clf.predict(test_representation) 
        print("Representation: ",name, "score", f1_score(y_test, test_predict))


if __name__ == '__main__':
    run()

```


In order to GridSearch the *SVD* representation
```python

for features in [10**3,25*10**2,5*10**3,10**4]:
    for dims in [128,256,512,1024,2048]:
        tmp_svd = SVD()
        r_train = tmp_svd.fit_transform(X_train,features,dims)
        ** YOUR CODE HERE **


```
