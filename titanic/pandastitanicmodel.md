
# Numpy


```python
import csv as csv
import numpy as np

csv_file_object = csv.reader(open('csv/train.csv','rb'))
header = csv_file_object.next()
data=[]

for row in csv_file_object:
    data.append(row)
data = np.array(data)

print(data)
```

    [['1' '0' '3' ..., '7.25' '' 'S']
     ['2' '1' '1' ..., '71.2833' 'C85' 'C']
     ['3' '1' '3' ..., '7.925' '' 'S']
     ..., 
     ['889' '0' '3' ..., '23.45' '' 'S']
     ['890' '1' '1' ..., '30' 'C148' 'C']
     ['891' '0' '3' ..., '7.75' '' 'Q']]



```python

```


```python
data[0:15,5]
```




    array(['22', '38', '26', '35', '35', '', '54', '2', '27', '14', '4', '58',
           '20', '39', '14'], 
          dtype='|S82')




```python
type(data[0::,5])
```




    numpy.ndarray




```python
ages_onboard = data[0::,5].astype(np.float)
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-13-c15380fd29f0> in <module>()
    ----> 1 ages_onboard = data[0::,5].astype(np.float)
    

    ValueError: could not convert string to float: 


# Playing with Pandas


```python
import pandas as pd

# Always use header = 0 when you know row 0 is the header row.
df = pd.read_csv('csv/train.csv',header=0)

df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>0</td>
      <td>3</td>
      <td>Moran, Mr. James</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>330877</td>
      <td>8.4583</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>0</td>
      <td>1</td>
      <td>McCarthy, Mr. Timothy J</td>
      <td>male</td>
      <td>54.0</td>
      <td>0</td>
      <td>0</td>
      <td>17463</td>
      <td>51.8625</td>
      <td>E46</td>
      <td>S</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>0</td>
      <td>3</td>
      <td>Palsson, Master. Gosta Leonard</td>
      <td>male</td>
      <td>2.0</td>
      <td>3</td>
      <td>1</td>
      <td>349909</td>
      <td>21.0750</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>1</td>
      <td>3</td>
      <td>Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)</td>
      <td>female</td>
      <td>27.0</td>
      <td>0</td>
      <td>2</td>
      <td>347742</td>
      <td>11.1333</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>1</td>
      <td>2</td>
      <td>Nasser, Mrs. Nicholas (Adele Achem)</td>
      <td>female</td>
      <td>14.0</td>
      <td>1</td>
      <td>0</td>
      <td>237736</td>
      <td>30.0708</td>
      <td>NaN</td>
      <td>C</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>1</td>
      <td>3</td>
      <td>Sandstrom, Miss. Marguerite Rut</td>
      <td>female</td>
      <td>4.0</td>
      <td>1</td>
      <td>1</td>
      <td>PP 9549</td>
      <td>16.7000</td>
      <td>G6</td>
      <td>S</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12</td>
      <td>1</td>
      <td>1</td>
      <td>Bonnell, Miss. Elizabeth</td>
      <td>female</td>
      <td>58.0</td>
      <td>0</td>
      <td>0</td>
      <td>113783</td>
      <td>26.5500</td>
      <td>C103</td>
      <td>S</td>
    </tr>
    <tr>
      <th>12</th>
      <td>13</td>
      <td>0</td>
      <td>3</td>
      <td>Saundercock, Mr. William Henry</td>
      <td>male</td>
      <td>20.0</td>
      <td>0</td>
      <td>0</td>
      <td>A/5. 2151</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>13</th>
      <td>14</td>
      <td>0</td>
      <td>3</td>
      <td>Andersson, Mr. Anders Johan</td>
      <td>male</td>
      <td>39.0</td>
      <td>1</td>
      <td>5</td>
      <td>347082</td>
      <td>31.2750</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>14</th>
      <td>15</td>
      <td>0</td>
      <td>3</td>
      <td>Vestrom, Miss. Hulda Amanda Adolfina</td>
      <td>female</td>
      <td>14.0</td>
      <td>0</td>
      <td>0</td>
      <td>350406</td>
      <td>7.8542</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>15</th>
      <td>16</td>
      <td>1</td>
      <td>2</td>
      <td>Hewlett, Mrs. (Mary D Kingcome)</td>
      <td>female</td>
      <td>55.0</td>
      <td>0</td>
      <td>0</td>
      <td>248706</td>
      <td>16.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>16</th>
      <td>17</td>
      <td>0</td>
      <td>3</td>
      <td>Rice, Master. Eugene</td>
      <td>male</td>
      <td>2.0</td>
      <td>4</td>
      <td>1</td>
      <td>382652</td>
      <td>29.1250</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>17</th>
      <td>18</td>
      <td>1</td>
      <td>2</td>
      <td>Williams, Mr. Charles Eugene</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>244373</td>
      <td>13.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>18</th>
      <td>19</td>
      <td>0</td>
      <td>3</td>
      <td>Vander Planke, Mrs. Julius (Emelia Maria Vande...</td>
      <td>female</td>
      <td>31.0</td>
      <td>1</td>
      <td>0</td>
      <td>345763</td>
      <td>18.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>19</th>
      <td>20</td>
      <td>1</td>
      <td>3</td>
      <td>Masselmani, Mrs. Fatima</td>
      <td>female</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>2649</td>
      <td>7.2250</td>
      <td>NaN</td>
      <td>C</td>
    </tr>
    <tr>
      <th>20</th>
      <td>21</td>
      <td>0</td>
      <td>2</td>
      <td>Fynney, Mr. Joseph J</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>239865</td>
      <td>26.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>21</th>
      <td>22</td>
      <td>1</td>
      <td>2</td>
      <td>Beesley, Mr. Lawrence</td>
      <td>male</td>
      <td>34.0</td>
      <td>0</td>
      <td>0</td>
      <td>248698</td>
      <td>13.0000</td>
      <td>D56</td>
      <td>S</td>
    </tr>
    <tr>
      <th>22</th>
      <td>23</td>
      <td>1</td>
      <td>3</td>
      <td>McGowan, Miss. Anna "Annie"</td>
      <td>female</td>
      <td>15.0</td>
      <td>0</td>
      <td>0</td>
      <td>330923</td>
      <td>8.0292</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>23</th>
      <td>24</td>
      <td>1</td>
      <td>1</td>
      <td>Sloper, Mr. William Thompson</td>
      <td>male</td>
      <td>28.0</td>
      <td>0</td>
      <td>0</td>
      <td>113788</td>
      <td>35.5000</td>
      <td>A6</td>
      <td>S</td>
    </tr>
    <tr>
      <th>24</th>
      <td>25</td>
      <td>0</td>
      <td>3</td>
      <td>Palsson, Miss. Torborg Danira</td>
      <td>female</td>
      <td>8.0</td>
      <td>3</td>
      <td>1</td>
      <td>349909</td>
      <td>21.0750</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>25</th>
      <td>26</td>
      <td>1</td>
      <td>3</td>
      <td>Asplund, Mrs. Carl Oscar (Selma Augusta Emilia...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>5</td>
      <td>347077</td>
      <td>31.3875</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>26</th>
      <td>27</td>
      <td>0</td>
      <td>3</td>
      <td>Emir, Mr. Farred Chehab</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>2631</td>
      <td>7.2250</td>
      <td>NaN</td>
      <td>C</td>
    </tr>
    <tr>
      <th>27</th>
      <td>28</td>
      <td>0</td>
      <td>1</td>
      <td>Fortune, Mr. Charles Alexander</td>
      <td>male</td>
      <td>19.0</td>
      <td>3</td>
      <td>2</td>
      <td>19950</td>
      <td>263.0000</td>
      <td>C23 C25 C27</td>
      <td>S</td>
    </tr>
    <tr>
      <th>28</th>
      <td>29</td>
      <td>1</td>
      <td>3</td>
      <td>O'Dwyer, Miss. Ellen "Nellie"</td>
      <td>female</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>330959</td>
      <td>7.8792</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>29</th>
      <td>30</td>
      <td>0</td>
      <td>3</td>
      <td>Todoroff, Mr. Lalio</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>349216</td>
      <td>7.8958</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>861</th>
      <td>862</td>
      <td>0</td>
      <td>2</td>
      <td>Giles, Mr. Frederick Edward</td>
      <td>male</td>
      <td>21.0</td>
      <td>1</td>
      <td>0</td>
      <td>28134</td>
      <td>11.5000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>862</th>
      <td>863</td>
      <td>1</td>
      <td>1</td>
      <td>Swift, Mrs. Frederick Joel (Margaret Welles Ba...</td>
      <td>female</td>
      <td>48.0</td>
      <td>0</td>
      <td>0</td>
      <td>17466</td>
      <td>25.9292</td>
      <td>D17</td>
      <td>S</td>
    </tr>
    <tr>
      <th>863</th>
      <td>864</td>
      <td>0</td>
      <td>3</td>
      <td>Sage, Miss. Dorothy Edith "Dolly"</td>
      <td>female</td>
      <td>NaN</td>
      <td>8</td>
      <td>2</td>
      <td>CA. 2343</td>
      <td>69.5500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>864</th>
      <td>865</td>
      <td>0</td>
      <td>2</td>
      <td>Gill, Mr. John William</td>
      <td>male</td>
      <td>24.0</td>
      <td>0</td>
      <td>0</td>
      <td>233866</td>
      <td>13.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>865</th>
      <td>866</td>
      <td>1</td>
      <td>2</td>
      <td>Bystrom, Mrs. (Karolina)</td>
      <td>female</td>
      <td>42.0</td>
      <td>0</td>
      <td>0</td>
      <td>236852</td>
      <td>13.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>866</th>
      <td>867</td>
      <td>1</td>
      <td>2</td>
      <td>Duran y More, Miss. Asuncion</td>
      <td>female</td>
      <td>27.0</td>
      <td>1</td>
      <td>0</td>
      <td>SC/PARIS 2149</td>
      <td>13.8583</td>
      <td>NaN</td>
      <td>C</td>
    </tr>
    <tr>
      <th>867</th>
      <td>868</td>
      <td>0</td>
      <td>1</td>
      <td>Roebling, Mr. Washington Augustus II</td>
      <td>male</td>
      <td>31.0</td>
      <td>0</td>
      <td>0</td>
      <td>PC 17590</td>
      <td>50.4958</td>
      <td>A24</td>
      <td>S</td>
    </tr>
    <tr>
      <th>868</th>
      <td>869</td>
      <td>0</td>
      <td>3</td>
      <td>van Melkebeke, Mr. Philemon</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>345777</td>
      <td>9.5000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>869</th>
      <td>870</td>
      <td>1</td>
      <td>3</td>
      <td>Johnson, Master. Harold Theodor</td>
      <td>male</td>
      <td>4.0</td>
      <td>1</td>
      <td>1</td>
      <td>347742</td>
      <td>11.1333</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>870</th>
      <td>871</td>
      <td>0</td>
      <td>3</td>
      <td>Balkic, Mr. Cerin</td>
      <td>male</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>349248</td>
      <td>7.8958</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>871</th>
      <td>872</td>
      <td>1</td>
      <td>1</td>
      <td>Beckwith, Mrs. Richard Leonard (Sallie Monypeny)</td>
      <td>female</td>
      <td>47.0</td>
      <td>1</td>
      <td>1</td>
      <td>11751</td>
      <td>52.5542</td>
      <td>D35</td>
      <td>S</td>
    </tr>
    <tr>
      <th>872</th>
      <td>873</td>
      <td>0</td>
      <td>1</td>
      <td>Carlsson, Mr. Frans Olof</td>
      <td>male</td>
      <td>33.0</td>
      <td>0</td>
      <td>0</td>
      <td>695</td>
      <td>5.0000</td>
      <td>B51 B53 B55</td>
      <td>S</td>
    </tr>
    <tr>
      <th>873</th>
      <td>874</td>
      <td>0</td>
      <td>3</td>
      <td>Vander Cruyssen, Mr. Victor</td>
      <td>male</td>
      <td>47.0</td>
      <td>0</td>
      <td>0</td>
      <td>345765</td>
      <td>9.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>874</th>
      <td>875</td>
      <td>1</td>
      <td>2</td>
      <td>Abelson, Mrs. Samuel (Hannah Wizosky)</td>
      <td>female</td>
      <td>28.0</td>
      <td>1</td>
      <td>0</td>
      <td>P/PP 3381</td>
      <td>24.0000</td>
      <td>NaN</td>
      <td>C</td>
    </tr>
    <tr>
      <th>875</th>
      <td>876</td>
      <td>1</td>
      <td>3</td>
      <td>Najib, Miss. Adele Kiamie "Jane"</td>
      <td>female</td>
      <td>15.0</td>
      <td>0</td>
      <td>0</td>
      <td>2667</td>
      <td>7.2250</td>
      <td>NaN</td>
      <td>C</td>
    </tr>
    <tr>
      <th>876</th>
      <td>877</td>
      <td>0</td>
      <td>3</td>
      <td>Gustafsson, Mr. Alfred Ossian</td>
      <td>male</td>
      <td>20.0</td>
      <td>0</td>
      <td>0</td>
      <td>7534</td>
      <td>9.8458</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>877</th>
      <td>878</td>
      <td>0</td>
      <td>3</td>
      <td>Petroff, Mr. Nedelio</td>
      <td>male</td>
      <td>19.0</td>
      <td>0</td>
      <td>0</td>
      <td>349212</td>
      <td>7.8958</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>878</th>
      <td>879</td>
      <td>0</td>
      <td>3</td>
      <td>Laleff, Mr. Kristo</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>349217</td>
      <td>7.8958</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>879</th>
      <td>880</td>
      <td>1</td>
      <td>1</td>
      <td>Potter, Mrs. Thomas Jr (Lily Alexenia Wilson)</td>
      <td>female</td>
      <td>56.0</td>
      <td>0</td>
      <td>1</td>
      <td>11767</td>
      <td>83.1583</td>
      <td>C50</td>
      <td>C</td>
    </tr>
    <tr>
      <th>880</th>
      <td>881</td>
      <td>1</td>
      <td>2</td>
      <td>Shelley, Mrs. William (Imanita Parrish Hall)</td>
      <td>female</td>
      <td>25.0</td>
      <td>0</td>
      <td>1</td>
      <td>230433</td>
      <td>26.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>881</th>
      <td>882</td>
      <td>0</td>
      <td>3</td>
      <td>Markun, Mr. Johann</td>
      <td>male</td>
      <td>33.0</td>
      <td>0</td>
      <td>0</td>
      <td>349257</td>
      <td>7.8958</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>882</th>
      <td>883</td>
      <td>0</td>
      <td>3</td>
      <td>Dahlberg, Miss. Gerda Ulrika</td>
      <td>female</td>
      <td>22.0</td>
      <td>0</td>
      <td>0</td>
      <td>7552</td>
      <td>10.5167</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>883</th>
      <td>884</td>
      <td>0</td>
      <td>2</td>
      <td>Banfield, Mr. Frederick James</td>
      <td>male</td>
      <td>28.0</td>
      <td>0</td>
      <td>0</td>
      <td>C.A./SOTON 34068</td>
      <td>10.5000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>884</th>
      <td>885</td>
      <td>0</td>
      <td>3</td>
      <td>Sutehall, Mr. Henry Jr</td>
      <td>male</td>
      <td>25.0</td>
      <td>0</td>
      <td>0</td>
      <td>SOTON/OQ 392076</td>
      <td>7.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>885</th>
      <td>886</td>
      <td>0</td>
      <td>3</td>
      <td>Rice, Mrs. William (Margaret Norton)</td>
      <td>female</td>
      <td>39.0</td>
      <td>0</td>
      <td>5</td>
      <td>382652</td>
      <td>29.1250</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>886</th>
      <td>887</td>
      <td>0</td>
      <td>2</td>
      <td>Montvila, Rev. Juozas</td>
      <td>male</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>211536</td>
      <td>13.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>887</th>
      <td>888</td>
      <td>1</td>
      <td>1</td>
      <td>Graham, Miss. Margaret Edith</td>
      <td>female</td>
      <td>19.0</td>
      <td>0</td>
      <td>0</td>
      <td>112053</td>
      <td>30.0000</td>
      <td>B42</td>
      <td>S</td>
    </tr>
    <tr>
      <th>888</th>
      <td>889</td>
      <td>0</td>
      <td>3</td>
      <td>Johnston, Miss. Catherine Helen "Carrie"</td>
      <td>female</td>
      <td>NaN</td>
      <td>1</td>
      <td>2</td>
      <td>W./C. 6607</td>
      <td>23.4500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>889</th>
      <td>890</td>
      <td>1</td>
      <td>1</td>
      <td>Behr, Mr. Karl Howell</td>
      <td>male</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>111369</td>
      <td>30.0000</td>
      <td>C148</td>
      <td>C</td>
    </tr>
    <tr>
      <th>890</th>
      <td>891</td>
      <td>0</td>
      <td>3</td>
      <td>Dooley, Mr. Patrick</td>
      <td>male</td>
      <td>32.0</td>
      <td>0</td>
      <td>0</td>
      <td>370376</td>
      <td>7.7500</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
  </tbody>
</table>
<p>891 rows × 12 columns</p>
</div>




```python
df.head(3)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.dtypes
```




    PassengerId      int64
    Survived         int64
    Pclass           int64
    Name            object
    Sex             object
    Age            float64
    SibSp            int64
    Parch            int64
    Ticket          object
    Fare           float64
    Cabin           object
    Embarked        object
    dtype: object




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 12 columns):
    PassengerId    891 non-null int64
    Survived       891 non-null int64
    Pclass         891 non-null int64
    Name           891 non-null object
    Sex            891 non-null object
    Age            714 non-null float64
    SibSp          891 non-null int64
    Parch          891 non-null int64
    Ticket         891 non-null object
    Fare           891 non-null float64
    Cabin          204 non-null object
    Embarked       889 non-null object
    dtypes: float64(2), int64(5), object(5)
    memory usage: 83.6+ KB



```python
df.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>714.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>446.000000</td>
      <td>0.383838</td>
      <td>2.308642</td>
      <td>29.699118</td>
      <td>0.523008</td>
      <td>0.381594</td>
      <td>32.204208</td>
    </tr>
    <tr>
      <th>std</th>
      <td>257.353842</td>
      <td>0.486592</td>
      <td>0.836071</td>
      <td>14.526497</td>
      <td>1.102743</td>
      <td>0.806057</td>
      <td>49.693429</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.420000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>223.500000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>20.125000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.910400</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>446.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>28.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>14.454200</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>668.500000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>38.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>31.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>891.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>80.000000</td>
      <td>8.000000</td>
      <td>6.000000</td>
      <td>512.329200</td>
    </tr>
  </tbody>
</table>
</div>



# Data Munging


```python
df['Age'][0:10]
```




    0    22.0
    1    38.0
    2    26.0
    3    35.0
    4    35.0
    5     NaN
    6    54.0
    7     2.0
    8    27.0
    9    14.0
    Name: Age, dtype: float64




```python
df['Cabin'][0:10]
```




    0     NaN
    1     C85
    2     NaN
    3    C123
    4     NaN
    5     NaN
    6     E46
    7     NaN
    8     NaN
    9     NaN
    Name: Cabin, dtype: object




```python
df['Age'].mean()
```




    29.69911764705882




```python
df[['Sex','Pclass','Age']]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sex</th>
      <th>Pclass</th>
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>male</td>
      <td>3</td>
      <td>22.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>female</td>
      <td>1</td>
      <td>38.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>female</td>
      <td>3</td>
      <td>26.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>female</td>
      <td>1</td>
      <td>35.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>male</td>
      <td>3</td>
      <td>35.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>male</td>
      <td>3</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>male</td>
      <td>1</td>
      <td>54.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>male</td>
      <td>3</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>female</td>
      <td>3</td>
      <td>27.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>female</td>
      <td>2</td>
      <td>14.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>female</td>
      <td>3</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>female</td>
      <td>1</td>
      <td>58.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>male</td>
      <td>3</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>male</td>
      <td>3</td>
      <td>39.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>female</td>
      <td>3</td>
      <td>14.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>female</td>
      <td>2</td>
      <td>55.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>male</td>
      <td>3</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>male</td>
      <td>2</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>18</th>
      <td>female</td>
      <td>3</td>
      <td>31.0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>female</td>
      <td>3</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20</th>
      <td>male</td>
      <td>2</td>
      <td>35.0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>male</td>
      <td>2</td>
      <td>34.0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>female</td>
      <td>3</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>male</td>
      <td>1</td>
      <td>28.0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>female</td>
      <td>3</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>female</td>
      <td>3</td>
      <td>38.0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>male</td>
      <td>3</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>27</th>
      <td>male</td>
      <td>1</td>
      <td>19.0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>female</td>
      <td>3</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>29</th>
      <td>male</td>
      <td>3</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>861</th>
      <td>male</td>
      <td>2</td>
      <td>21.0</td>
    </tr>
    <tr>
      <th>862</th>
      <td>female</td>
      <td>1</td>
      <td>48.0</td>
    </tr>
    <tr>
      <th>863</th>
      <td>female</td>
      <td>3</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>864</th>
      <td>male</td>
      <td>2</td>
      <td>24.0</td>
    </tr>
    <tr>
      <th>865</th>
      <td>female</td>
      <td>2</td>
      <td>42.0</td>
    </tr>
    <tr>
      <th>866</th>
      <td>female</td>
      <td>2</td>
      <td>27.0</td>
    </tr>
    <tr>
      <th>867</th>
      <td>male</td>
      <td>1</td>
      <td>31.0</td>
    </tr>
    <tr>
      <th>868</th>
      <td>male</td>
      <td>3</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>869</th>
      <td>male</td>
      <td>3</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>870</th>
      <td>male</td>
      <td>3</td>
      <td>26.0</td>
    </tr>
    <tr>
      <th>871</th>
      <td>female</td>
      <td>1</td>
      <td>47.0</td>
    </tr>
    <tr>
      <th>872</th>
      <td>male</td>
      <td>1</td>
      <td>33.0</td>
    </tr>
    <tr>
      <th>873</th>
      <td>male</td>
      <td>3</td>
      <td>47.0</td>
    </tr>
    <tr>
      <th>874</th>
      <td>female</td>
      <td>2</td>
      <td>28.0</td>
    </tr>
    <tr>
      <th>875</th>
      <td>female</td>
      <td>3</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>876</th>
      <td>male</td>
      <td>3</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>877</th>
      <td>male</td>
      <td>3</td>
      <td>19.0</td>
    </tr>
    <tr>
      <th>878</th>
      <td>male</td>
      <td>3</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>879</th>
      <td>female</td>
      <td>1</td>
      <td>56.0</td>
    </tr>
    <tr>
      <th>880</th>
      <td>female</td>
      <td>2</td>
      <td>25.0</td>
    </tr>
    <tr>
      <th>881</th>
      <td>male</td>
      <td>3</td>
      <td>33.0</td>
    </tr>
    <tr>
      <th>882</th>
      <td>female</td>
      <td>3</td>
      <td>22.0</td>
    </tr>
    <tr>
      <th>883</th>
      <td>male</td>
      <td>2</td>
      <td>28.0</td>
    </tr>
    <tr>
      <th>884</th>
      <td>male</td>
      <td>3</td>
      <td>25.0</td>
    </tr>
    <tr>
      <th>885</th>
      <td>female</td>
      <td>3</td>
      <td>39.0</td>
    </tr>
    <tr>
      <th>886</th>
      <td>male</td>
      <td>2</td>
      <td>27.0</td>
    </tr>
    <tr>
      <th>887</th>
      <td>female</td>
      <td>1</td>
      <td>19.0</td>
    </tr>
    <tr>
      <th>888</th>
      <td>female</td>
      <td>3</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>889</th>
      <td>male</td>
      <td>1</td>
      <td>26.0</td>
    </tr>
    <tr>
      <th>890</th>
      <td>male</td>
      <td>3</td>
      <td>32.0</td>
    </tr>
  </tbody>
</table>
<p>891 rows × 3 columns</p>
</div>




```python
df[df['Age']>60][['Sex','Pclass','Age','Survived']]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sex</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>33</th>
      <td>male</td>
      <td>2</td>
      <td>66.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>54</th>
      <td>male</td>
      <td>1</td>
      <td>65.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>96</th>
      <td>male</td>
      <td>1</td>
      <td>71.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>116</th>
      <td>male</td>
      <td>3</td>
      <td>70.5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>170</th>
      <td>male</td>
      <td>1</td>
      <td>61.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>252</th>
      <td>male</td>
      <td>1</td>
      <td>62.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>275</th>
      <td>female</td>
      <td>1</td>
      <td>63.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>280</th>
      <td>male</td>
      <td>3</td>
      <td>65.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>326</th>
      <td>male</td>
      <td>3</td>
      <td>61.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>438</th>
      <td>male</td>
      <td>1</td>
      <td>64.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>456</th>
      <td>male</td>
      <td>1</td>
      <td>65.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>483</th>
      <td>female</td>
      <td>3</td>
      <td>63.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>493</th>
      <td>male</td>
      <td>1</td>
      <td>71.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>545</th>
      <td>male</td>
      <td>1</td>
      <td>64.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>555</th>
      <td>male</td>
      <td>1</td>
      <td>62.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>570</th>
      <td>male</td>
      <td>2</td>
      <td>62.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>625</th>
      <td>male</td>
      <td>1</td>
      <td>61.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>630</th>
      <td>male</td>
      <td>1</td>
      <td>80.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>672</th>
      <td>male</td>
      <td>2</td>
      <td>70.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>745</th>
      <td>male</td>
      <td>1</td>
      <td>70.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>829</th>
      <td>female</td>
      <td>1</td>
      <td>62.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>851</th>
      <td>male</td>
      <td>3</td>
      <td>74.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[df['Age'].isnull()][['Sex','Pclass','Age','Survived']]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sex</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>male</td>
      <td>3</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>male</td>
      <td>2</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
    <tr>
      <th>19</th>
      <td>female</td>
      <td>3</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
    <tr>
      <th>26</th>
      <td>male</td>
      <td>3</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>female</td>
      <td>3</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
    <tr>
      <th>29</th>
      <td>male</td>
      <td>3</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>31</th>
      <td>female</td>
      <td>1</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
    <tr>
      <th>32</th>
      <td>female</td>
      <td>3</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
    <tr>
      <th>36</th>
      <td>male</td>
      <td>3</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
    <tr>
      <th>42</th>
      <td>male</td>
      <td>3</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>45</th>
      <td>male</td>
      <td>3</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>46</th>
      <td>male</td>
      <td>3</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>47</th>
      <td>female</td>
      <td>3</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
    <tr>
      <th>48</th>
      <td>male</td>
      <td>3</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>55</th>
      <td>male</td>
      <td>1</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
    <tr>
      <th>64</th>
      <td>male</td>
      <td>1</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>65</th>
      <td>male</td>
      <td>3</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
    <tr>
      <th>76</th>
      <td>male</td>
      <td>3</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>77</th>
      <td>male</td>
      <td>3</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>82</th>
      <td>female</td>
      <td>3</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
    <tr>
      <th>87</th>
      <td>male</td>
      <td>3</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>95</th>
      <td>male</td>
      <td>3</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>101</th>
      <td>male</td>
      <td>3</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>107</th>
      <td>male</td>
      <td>3</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
    <tr>
      <th>109</th>
      <td>female</td>
      <td>3</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
    <tr>
      <th>121</th>
      <td>male</td>
      <td>3</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>126</th>
      <td>male</td>
      <td>3</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>128</th>
      <td>female</td>
      <td>3</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
    <tr>
      <th>140</th>
      <td>female</td>
      <td>3</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>154</th>
      <td>male</td>
      <td>3</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>718</th>
      <td>male</td>
      <td>3</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>727</th>
      <td>female</td>
      <td>3</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
    <tr>
      <th>732</th>
      <td>male</td>
      <td>2</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>738</th>
      <td>male</td>
      <td>3</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>739</th>
      <td>male</td>
      <td>3</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>740</th>
      <td>male</td>
      <td>1</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
    <tr>
      <th>760</th>
      <td>male</td>
      <td>3</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>766</th>
      <td>male</td>
      <td>1</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>768</th>
      <td>male</td>
      <td>3</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>773</th>
      <td>male</td>
      <td>3</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>776</th>
      <td>male</td>
      <td>3</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>778</th>
      <td>male</td>
      <td>3</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>783</th>
      <td>male</td>
      <td>3</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>790</th>
      <td>male</td>
      <td>3</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>792</th>
      <td>female</td>
      <td>3</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>793</th>
      <td>male</td>
      <td>1</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>815</th>
      <td>male</td>
      <td>1</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>825</th>
      <td>male</td>
      <td>3</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>826</th>
      <td>male</td>
      <td>3</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>828</th>
      <td>male</td>
      <td>3</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
    <tr>
      <th>832</th>
      <td>male</td>
      <td>3</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>837</th>
      <td>male</td>
      <td>3</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>839</th>
      <td>male</td>
      <td>1</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
    <tr>
      <th>846</th>
      <td>male</td>
      <td>3</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>849</th>
      <td>female</td>
      <td>1</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
    <tr>
      <th>859</th>
      <td>male</td>
      <td>3</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>863</th>
      <td>female</td>
      <td>3</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>868</th>
      <td>male</td>
      <td>3</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>878</th>
      <td>male</td>
      <td>3</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>888</th>
      <td>female</td>
      <td>3</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>177 rows × 4 columns</p>
</div>




```python
for i in range(1,4):
    print i, len(df[(df['Sex']=='male') & (df['Pclass']==i)])
```

    1 122
    2 108
    3 347



```python
import pylab as P
df['Age'].hist(bins=16,range=(0,80),alpha=.5)
P.show()
```


![png](output_20_0.png)


# Cleaning the Data


```python
df['Gender']= df['Sex'].map({'female':0, 'male':1}).astype(int)
df.head(3)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Gender</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
median_ages = np.zeros((2,3))
median_ages
```




    array([[ 0.,  0.,  0.],
           [ 0.,  0.,  0.]])




```python
for i in range(0,2):
    for j in range(0,3):
        median_ages[i,j] = df[(df['Gender'] == i) & \
                              (df['Pclass'] == j+1)]['Age'].dropna().median()
median_ages
```




    array([[ 35. ,  28. ,  21.5],
           [ 40. ,  30. ,  25. ]])




```python
df['AgeFill'] = df['Age']
df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Gender</th>
      <th>AgeFill</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
      <td>1</td>
      <td>22.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
      <td>0</td>
      <td>38.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
      <td>0</td>
      <td>26.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
      <td>0</td>
      <td>35.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
      <td>1</td>
      <td>35.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[df['Age'].isnull()][['Gender','Pclass','Age','AgeFill']].head(10)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gender</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>AgeFill</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>3</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1</td>
      <td>2</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0</td>
      <td>3</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>26</th>
      <td>1</td>
      <td>3</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>28</th>
      <td>0</td>
      <td>3</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>29</th>
      <td>1</td>
      <td>3</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>31</th>
      <td>0</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>32</th>
      <td>0</td>
      <td>3</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>36</th>
      <td>1</td>
      <td>3</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>42</th>
      <td>1</td>
      <td>3</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
for i in range(0,2):
    for j in range(0,3):
        df.loc[(df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1), \
                'AgeFill'] = median_ages[i,j]
df[df['Age'].isnull()][['Gender','Pclass','Age','AgeFill']].head(10)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gender</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>AgeFill</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>3</td>
      <td>NaN</td>
      <td>25.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1</td>
      <td>2</td>
      <td>NaN</td>
      <td>30.0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0</td>
      <td>3</td>
      <td>NaN</td>
      <td>21.5</td>
    </tr>
    <tr>
      <th>26</th>
      <td>1</td>
      <td>3</td>
      <td>NaN</td>
      <td>25.0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>0</td>
      <td>3</td>
      <td>NaN</td>
      <td>21.5</td>
    </tr>
    <tr>
      <th>29</th>
      <td>1</td>
      <td>3</td>
      <td>NaN</td>
      <td>25.0</td>
    </tr>
    <tr>
      <th>31</th>
      <td>0</td>
      <td>1</td>
      <td>NaN</td>
      <td>35.0</td>
    </tr>
    <tr>
      <th>32</th>
      <td>0</td>
      <td>3</td>
      <td>NaN</td>
      <td>21.5</td>
    </tr>
    <tr>
      <th>36</th>
      <td>1</td>
      <td>3</td>
      <td>NaN</td>
      <td>25.0</td>
    </tr>
    <tr>
      <th>42</th>
      <td>1</td>
      <td>3</td>
      <td>NaN</td>
      <td>25.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['AgeIsNull'] = pd.isnull(df.Age).astype(int)
df.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Gender</th>
      <th>AgeFill</th>
      <th>AgeIsNull</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>714.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>446.000000</td>
      <td>0.383838</td>
      <td>2.308642</td>
      <td>29.699118</td>
      <td>0.523008</td>
      <td>0.381594</td>
      <td>32.204208</td>
      <td>0.647587</td>
      <td>29.112424</td>
      <td>0.198653</td>
    </tr>
    <tr>
      <th>std</th>
      <td>257.353842</td>
      <td>0.486592</td>
      <td>0.836071</td>
      <td>14.526497</td>
      <td>1.102743</td>
      <td>0.806057</td>
      <td>49.693429</td>
      <td>0.477990</td>
      <td>13.304424</td>
      <td>0.399210</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.420000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.420000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>223.500000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>20.125000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.910400</td>
      <td>0.000000</td>
      <td>21.500000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>446.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>28.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>14.454200</td>
      <td>1.000000</td>
      <td>26.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>668.500000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>38.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>31.000000</td>
      <td>1.000000</td>
      <td>36.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>891.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>80.000000</td>
      <td>8.000000</td>
      <td>6.000000</td>
      <td>512.329200</td>
      <td>1.000000</td>
      <td>80.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



# Feature Engineering


```python
df['FamilySize'] = df['SibSp'] + df['Parch']
df['FamilySize'].plot.hist()
df.plot.scatter( x = 'Pclass', y = 'FamilySize')
P.show()
```


![png](output_30_0.png)



![png](output_30_1.png)



```python
df['Age*Class'] = df.AgeFill * df.Pclass
df['Age*Class'].hist()
P.show()
```


![png](output_31_0.png)


# Final preparation


```python
df.dtypes
```




    PassengerId      int64
    Survived         int64
    Pclass           int64
    Name            object
    Sex             object
    Age            float64
    SibSp            int64
    Parch            int64
    Ticket          object
    Fare           float64
    Cabin           object
    Embarked        object
    Gender           int64
    AgeFill        float64
    AgeIsNull        int64
    FamilySize       int64
    Age*Class      float64
    dtype: object




```python
df.dtypes[df.dtypes.map(lambda x: x=='object')]
```




    Name        object
    Sex         object
    Ticket      object
    Cabin       object
    Embarked    object
    dtype: object




```python
df = df.dropna()
```


```python
train_data = df.values
train_data
```




    array([[   1.,    0.,    3., ...,    0.,    1.,   66.],
           [   2.,    1.,    1., ...,    0.,    1.,   38.],
           [   3.,    1.,    3., ...,    0.,    0.,   78.],
           ..., 
           [ 888.,    1.,    1., ...,    0.,    0.,   19.],
           [ 890.,    1.,    1., ...,    0.,    0.,   26.],
           [ 891.,    0.,    3., ...,    0.,    0.,   96.]])




```python
data
```




    array([['1', '0', '3', ..., '7.25', '', 'S'],
           ['2', '1', '1', ..., '71.2833', 'C85', 'C'],
           ['3', '1', '3', ..., '7.925', '', 'S'],
           ..., 
           ['889', '0', '3', ..., '23.45', '', 'S'],
           ['890', '1', '1', ..., '30', 'C148', 'C'],
           ['891', '0', '3', ..., '7.75', '', 'Q']], 
          dtype='|S82')




```python

```
