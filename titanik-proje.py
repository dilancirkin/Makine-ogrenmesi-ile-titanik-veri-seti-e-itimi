{
 "cells": [
  {
   "cell_type": "markdown",
   "execution_count": null,
   "metadata": {
    "papermill": {
     "duration": 0.036075,
     "end_time": "2020-09-08T17:54:22.852931",
     "exception": false,
     "start_time": "2020-09-08T17:54:22.816856",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "# introduction (giriş kısmı)\n",
    "The sinking of Titanic is one of the most notorious shipwredcks is the history.In 1912,druing her voyage, the Titanic sank after colliding with an iceberg,killing 1502 out of 2224 passengers and crew.\n",
    "Titanik kötü ve ünlü kazalardan biridir.Buz dağıyla çarpıştııktan sonra titanik battı.Gemide yaklaşık 2224 kişi vardı.\n",
    "\n",
    "Content(İçerik):\n",
    "1.İlk olarak veri setimi yükleyeceğim ve içinde neler var bakacağım.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "_cell_guid": "b1076dfc-b9ad-4769-8c92-a6c4dae69d19",
    "_uuid": "8f2839f25d086af736a60e9eeb907d3b93b6e0e5",
    "execution": {
     "iopub.execute_input": "2020-09-08T17:54:22.933667Z",
     "iopub.status.busy": "2020-09-08T17:54:22.932838Z",
     "iopub.status.idle": "2020-09-08T17:54:23.925810Z",
     "shell.execute_reply": "2020-09-08T17:54:23.924692Z"
    },
    "papermill": {
     "duration": 1.037716,
     "end_time": "2020-09-08T17:54:23.925989",
     "exception": false,
     "start_time": "2020-09-08T17:54:22.888273",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "/kaggle/input/titanic/train.csv\n",
      "/kaggle/input/titanic/gender_submission.csv\n",
      "/kaggle/input/titanic/test.csv\n"
     ]
    }
   ],
   "source": [
    "\n",
    "import numpy as np # linear algebra\n",
    "import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)\n",
    "\n",
    "import matplotlib.pyplot as plt\n",
    "plt.style.use(\"seaborn-whitegrid\")\n",
    "\n",
    "import seaborn as sns\n",
    "\n",
    "from collections import Counter\n",
    "\n",
    "import warnings\n",
    "warnings.filterwarnings(\"ignore\") #pythondan kaynaklı uyarıları kapat\n",
    "\n",
    "import os\n",
    "for dirname, _, filenames in os.walk('/kaggle/input'):\n",
    "    for filename in filenames:\n",
    "        print(os.path.join(dirname, filename))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2020-09-08T17:54:24.007257Z",
     "iopub.status.busy": "2020-09-08T17:54:24.006415Z",
     "iopub.status.idle": "2020-09-08T17:54:24.065199Z",
     "shell.execute_reply": "2020-09-08T17:54:24.065789Z"
    },
    "papermill": {
     "duration": 0.105282,
     "end_time": "2020-09-08T17:54:24.065987",
     "exception": false,
     "start_time": "2020-09-08T17:54:23.960705",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "train_df = pd.read_csv(\"/kaggle/input/titanic/train.csv\")\n",
    "test_df = pd.read_csv(\"/kaggle/input/titanic/test.csv\")\n",
    "test_PassengerId = test_df[\"PassengerId\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2020-09-08T17:54:24.145456Z",
     "iopub.status.busy": "2020-09-08T17:54:24.144402Z",
     "iopub.status.idle": "2020-09-08T17:54:24.149145Z",
     "shell.execute_reply": "2020-09-08T17:54:24.148538Z"
    },
    "papermill": {
     "duration": 0.046776,
     "end_time": "2020-09-08T17:54:24.149270",
     "exception": false,
     "start_time": "2020-09-08T17:54:24.102494",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',\n",
       "       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],\n",
       "      dtype='object')"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_df.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2020-09-08T17:54:24.234559Z",
     "iopub.status.busy": "2020-09-08T17:54:24.233764Z",
     "iopub.status.idle": "2020-09-08T17:54:24.247176Z",
     "shell.execute_reply": "2020-09-08T17:54:24.246443Z"
    },
    "papermill": {
     "duration": 0.063526,
     "end_time": "2020-09-08T17:54:24.247301",
     "exception": false,
     "start_time": "2020-09-08T17:54:24.183775",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Name</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>Embarked</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Braund, Mr. Owen Harris</td>\n",
       "      <td>male</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>A/5 21171</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>\n",
       "      <td>female</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>PC 17599</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>C85</td>\n",
       "      <td>C</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>Heikkinen, Miss. Laina</td>\n",
       "      <td>female</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>STON/O2. 3101282</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>\n",
       "      <td>female</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>113803</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>C123</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Allen, Mr. William Henry</td>\n",
       "      <td>male</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>373450</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   PassengerId  Survived  Pclass  \\\n",
       "0            1         0       3   \n",
       "1            2         1       1   \n",
       "2            3         1       3   \n",
       "3            4         1       1   \n",
       "4            5         0       3   \n",
       "\n",
       "                                                Name     Sex   Age  SibSp  \\\n",
       "0                            Braund, Mr. Owen Harris    male  22.0      1   \n",
       "1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   \n",
       "2                             Heikkinen, Miss. Laina  female  26.0      0   \n",
       "3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   \n",
       "4                           Allen, Mr. William Henry    male  35.0      0   \n",
       "\n",
       "   Parch            Ticket     Fare Cabin Embarked  \n",
       "0      0         A/5 21171   7.2500   NaN        S  \n",
       "1      0          PC 17599  71.2833   C85        C  \n",
       "2      0  STON/O2. 3101282   7.9250   NaN        S  \n",
       "3      0            113803  53.1000  C123        S  \n",
       "4      0            373450   8.0500   NaN        S  "
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2020-09-08T17:54:24.334709Z",
     "iopub.status.busy": "2020-09-08T17:54:24.333936Z",
     "iopub.status.idle": "2020-09-08T17:54:24.367506Z",
     "shell.execute_reply": "2020-09-08T17:54:24.368400Z"
    },
    "papermill": {
     "duration": 0.086258,
     "end_time": "2020-09-08T17:54:24.368574",
     "exception": false,
     "start_time": "2020-09-08T17:54:24.282316",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>714.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>446.000000</td>\n",
       "      <td>0.383838</td>\n",
       "      <td>2.308642</td>\n",
       "      <td>29.699118</td>\n",
       "      <td>0.523008</td>\n",
       "      <td>0.381594</td>\n",
       "      <td>32.204208</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>257.353842</td>\n",
       "      <td>0.486592</td>\n",
       "      <td>0.836071</td>\n",
       "      <td>14.526497</td>\n",
       "      <td>1.102743</td>\n",
       "      <td>0.806057</td>\n",
       "      <td>49.693429</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.420000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>223.500000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>20.125000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>7.910400</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>446.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>28.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>14.454200</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>668.500000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>38.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>31.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>891.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>80.000000</td>\n",
       "      <td>8.000000</td>\n",
       "      <td>6.000000</td>\n",
       "      <td>512.329200</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       PassengerId    Survived      Pclass         Age       SibSp  \\\n",
       "count   891.000000  891.000000  891.000000  714.000000  891.000000   \n",
       "mean    446.000000    0.383838    2.308642   29.699118    0.523008   \n",
       "std     257.353842    0.486592    0.836071   14.526497    1.102743   \n",
       "min       1.000000    0.000000    1.000000    0.420000    0.000000   \n",
       "25%     223.500000    0.000000    2.000000   20.125000    0.000000   \n",
       "50%     446.000000    0.000000    3.000000   28.000000    0.000000   \n",
       "75%     668.500000    1.000000    3.000000   38.000000    1.000000   \n",
       "max     891.000000    1.000000    3.000000   80.000000    8.000000   \n",
       "\n",
       "            Parch        Fare  \n",
       "count  891.000000  891.000000  \n",
       "mean     0.381594   32.204208  \n",
       "std      0.806057   49.693429  \n",
       "min      0.000000    0.000000  \n",
       "25%      0.000000    7.910400  \n",
       "50%      0.000000   14.454200  \n",
       "75%      0.000000   31.000000  \n",
       "max      6.000000  512.329200  "
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_df.describe()"
   ]
  },
  {
   "cell_type": "markdown",
   "execution_count": null,
   "metadata": {
    "papermill": {
     "duration": 0.035229,
     "end_time": "2020-09-08T17:54:24.439349",
     "exception": false,
     "start_time": "2020-09-08T17:54:24.404120",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "Variable Description(Değişkenlerin tanımlanması)\n",
    "PassengerId:Bulunan yolcuların yolcu numarası.\n",
    "Survived:İçerisinde 0 ve 1 leri barındırır.0lar yolcunun öldüğünü,1ler hayatta kaldıpını gösterir.\n",
    "Pclass:1. sınıf,2. sınıf ve 3.sınıf yolcuların sınıflarını belirtir.\n",
    "Name:Yolcuların isimlerini belirtir.\n",
    "Sex:Yolcuların cinsiyetini belirtir.\n",
    "Age:Yolcuların yaşını belirtir.\n",
    "SibSp:Siblings:Kardşler sp:spoues(Eş) yani gemide akrabası var onu gösteriyoruz\n",
    "Parch:par:parent ch:children bu gemide çocuğu ya da ebeveyni var.\n",
    "Ticket:Bilet numarası.\n",
    "Fare:Bileti almak için ödediğimiz para miktarı.\n",
    "Cabin:Gemi içerisinde kaldığımız odanın numarası.\n",
    "Embarked:3 tane liman var.Gemiye hangi limandan bindiğimiz.(C,Q,S)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2020-09-08T17:54:24.526235Z",
     "iopub.status.busy": "2020-09-08T17:54:24.525270Z",
     "iopub.status.idle": "2020-09-08T17:54:24.528773Z",
     "shell.execute_reply": "2020-09-08T17:54:24.529331Z"
    },
    "papermill": {
     "duration": 0.054283,
     "end_time": "2020-09-08T17:54:24.529477",
     "exception": false,
     "start_time": "2020-09-08T17:54:24.475194",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 891 entries, 0 to 890\n",
      "Data columns (total 12 columns):\n",
      " #   Column       Non-Null Count  Dtype  \n",
      "---  ------       --------------  -----  \n",
      " 0   PassengerId  891 non-null    int64  \n",
      " 1   Survived     891 non-null    int64  \n",
      " 2   Pclass       891 non-null    int64  \n",
      " 3   Name         891 non-null    object \n",
      " 4   Sex          891 non-null    object \n",
      " 5   Age          714 non-null    float64\n",
      " 6   SibSp        891 non-null    int64  \n",
      " 7   Parch        891 non-null    int64  \n",
      " 8   Ticket       891 non-null    object \n",
      " 9   Fare         891 non-null    float64\n",
      " 10  Cabin        204 non-null    object \n",
      " 11  Embarked     889 non-null    object \n",
      "dtypes: float64(2), int64(5), object(5)\n",
      "memory usage: 83.7+ KB\n"
     ]
    }
   ],
   "source": [
    "train_df.info()"
   ]
  },
  {
   "cell_type": "markdown",
   "execution_count": null,
   "metadata": {
    "papermill": {
     "duration": 0.03508,
     "end_time": "2020-09-08T17:54:24.599914",
     "exception": false,
     "start_time": "2020-09-08T17:54:24.564834",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "float64(2):yolvu yaşları ve paralar\n",
    "int64(5):pclass,sipsb,parch,passengerıd and survived\n",
    "object(5):cabin,embarked,name,sex,ticket"
   ]
  },
  {
   "cell_type": "markdown",
   "execution_count": null,
   "metadata": {
    "papermill": {
     "duration": 0.035062,
     "end_time": "2020-09-08T17:54:24.670869",
     "exception": false,
     "start_time": "2020-09-08T17:54:24.635807",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "# Univariate Variable Analysis\n",
    "Categorical Variable:İki veya daha fazla kategorşden oluşan değişkenler.Örneğin cinsiyet erkek ve kadın.sırvived,sex,pclass,embarked,name,cabin,ticket,sibsp,parch.Kaç tane kategori olduğunu öğrenmek için bunun analizini yaparız.\n",
    "Numerical Variable:Fare,age and passengerId.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2020-09-08T17:54:24.765302Z",
     "iopub.status.busy": "2020-09-08T17:54:24.764425Z",
     "iopub.status.idle": "2020-09-08T17:54:24.768360Z",
     "shell.execute_reply": "2020-09-08T17:54:24.767701Z"
    },
    "papermill": {
     "duration": 0.062198,
     "end_time": "2020-09-08T17:54:24.768482",
     "exception": false,
     "start_time": "2020-09-08T17:54:24.706284",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "def bar_plot(variable):\n",
    "    \"\"\"input:variable ex:\"Sex\"\n",
    "       output:bar plot& value count\n",
    "    \"\"\"\n",
    "    var = train_df[variable]\n",
    "    varValue = var.value_counts()\n",
    "    \n",
    "    #visualize(görselleştirme)\n",
    "    plt.figure(figsize = (9,3))\n",
    "    plt.bar(varValue.index, varValue)\n",
    "    plt.xticks(varValue.index, varValue.index.values)\n",
    "    plt.ylabel(\"Frequency\")\n",
    "    plt.title(variable)\n",
    "    plt.show()\n",
    "    print(\"{}: \\n {}\".format(variable,varValue))\n",
    "    \n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2020-09-08T17:54:24.847157Z",
     "iopub.status.busy": "2020-09-08T17:54:24.846306Z",
     "iopub.status.idle": "2020-09-08T17:54:25.725916Z",
     "shell.execute_reply": "2020-09-08T17:54:25.724723Z"
    },
    "papermill": {
     "duration": 0.9218,
     "end_time": "2020-09-08T17:54:25.726094",
     "exception": false,
     "start_time": "2020-09-08T17:54:24.804294",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAiYAAADNCAYAAACM0rsuAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjEsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+j8jraAAAXN0lEQVR4nO3de1BU5/3H8c8qbEHdirgLVmwbp9qRKmAUnWhCTLooq07rJpri0JrWW2Olmhpab7ENqY5VjI2XME3HTPDWNIxbJyUTK9RYZ7wgjWyk0pCq0TSNQdj1hhFQwf39kXF/MV6yJhz34L5f/7Dn4Zxnv8fhcT7zPGeftQQCgYAAAABMoEO4CwAAALiKYAIAAEyDYAIAAEyDYAIAAEyDYAIAAEyDYAIAAEyDYALgC6murtaPf/xjuVwuZWVlKTs7WwcOHGiTvleuXKk///nPbdLXX//6V02aNKlN+gJgvKhwFwCg/QkEApoxY4aWLFmihx56SJJUVlam3Nxc7dq1S7GxsV+q/7y8vDaoEkB7RDABcNvOnDkjn8+ntLS0YNuoUaOUmpqqv/3tbyopKdH69eslSVu3bg0ez58/X127dtW+ffuUlZWljRs3at++fYqK+uS/op/97Gd68MEHVVVVpW984xs6f/68Ll++rEWLFgXf9+GHH9bu3btVV1en/Px8+Xw+Wa1WLV26VCkpKbpy5YqWLFminTt3ym63a8iQIXf83wfAF8dSDoDb1q1bN6WkpOjxxx/Xli1b9OGHH0qSevTo8bnXlpeXy+Px6Oc//7nsdntw+aepqUkVFRXKysoKnutyubRz587g8c6dO3Xfffepc+fOmjNnjsaNG6fS0lLl5+dr5syZamlp0e7du7V371698cYb2rx5c5stLwG4MwgmAG6bxWJRUVGRRo4cqY0bN8rpdGrs2LEqKyv73GuHDRumr3zlK5I+mWW5Gjx2796t1NRUxcfHB89NS0tTIBDQu+++K0n6+9//rtGjR+vYsWP64IMPNH78eEnS4MGDFR8fr7fffltvvfWWRowYoc6dOysmJkajR49u69sHYCCCCYAvxGazafbs2Xr99de1d+9ejRs3Tk899ZSam5tveV3Xrl2Drz89I7Jjxw6NGTPmuvNHjhypN998U42NjfJ6vXI6nWpoaFBra6vGjBkjl8sll8ulU6dO6ezZszp37pxsNlvw+q9+9attdMcA7gSeMQFw206ePKkPP/xQ6enpkiS73a6f/vSn2r59u2JjY9Xa2ho899y5czftp1+/furYsaPeffdd7dmzRwsWLLjunKysLC1dulR9+/bVkCFD1KVLFyUkJKhz587avn37decfPHhQ58+fDx6fPn36y9wqgDuMGRMAt622tla5ubmqrq4Otv3rX//SRx99JEl6//33dfHiRTU1Nam0tPSWfY0aNUpr165VcnKyunXrdt3vBw0apFOnTmnr1q3BZZmkpCT16NEjGExOnz6tp556So2Njbr33nu1Z88eNTc3q6mp6YbhBYB5MWMC4Lbde++9Wrx4sfLz83X+/HlduXJF3bt31/PPP6+hQ4dqx44dysrKUq9evZSZmak9e/bctC+Xy6VHH31US5YsueHvLRaLMjMztWXLFq1cuTLY9vvf/175+flatWqVOnTooMmTJ6tTp056+OGHtWvXLmVlZclut2vEiBE8AAu0I5ZAIBAIdxEAAAASSzkAAMBECCYAAMA0CCYAAMA0CCYAAMA02s2nciorK8NdAgAAaEODBw++rq3dBBPpxjeAu1NNTY2Sk5PDXQaAO4QxH3luNuHAUg4AADANggkAADANggkAADANggkAADANggkAADANggkAADCNdvVxYSPcM/+NcJeAmzoW7gLwKe8vGxvuEgBEAGZMAACAaRBMAACAaRBMAACAaRBMAACAaRBMAACAaRBMAACAaRBMAACAaRBMAACAaRBMAACAaRBMAACAaRi2JX11dbVmzpypb37zm5Kkb3/725o2bZrmzp2r1tZWORwOrVixQlarVSUlJdqwYYM6dOig7OxsTZgwwaiyAACAiRkWTBobG5WVlaWnn3462LZgwQLl5ORo9OjRKigokMfjkdvtVmFhoTwej6Kjo+V2u5WZmam4uDijSgMAACZl2FLOhQsXrmurqKiQ0+mUJDmdTpWXl6uqqkopKSmy2WyKiYlRenq6vF6vUWUBAAATM3TGpLKyUtOmTVNTU5NmzZqlpqYmWa1WSZLD4ZDP55Pf71d8fHzwOrvdLp/Pd8M+a2pqjCoXwOdg/MFIzc3N/I1BkoHBpF+/fsrNzZXT6dTx48c1efJktbS0BH8fCASu+fnpdovFcsM+k5OTDaj0mAF9AncfY8Yf8Imamhr+xiJMZWXlDdsNW8r51re+FVy26d27t+x2uxoaGtTc3CxJqqurU0JCghITE+X3+4PX1dfXy+FwGFUWAAAwMcOCicfj0caNGyVJPp9Pp06d0qOPPqrS0lJJUllZmTIyMpSWlqZDhw6poaFBFy5ckNfrVXp6ulFlAQAAEzNsKWfkyJH65S9/qdLSUl26dEn5+flKTk7WvHnzVFxcrJ49e8rtdis6Olp5eXmaOnWqLBaLcnNzZbPZjCoLAACYmGHBpGvXrlq3bt117UVFRde1uVwuuVwuo0oBAADtBDu/AgAA0yCYAAAA0yCYAAAA0yCYAAAA0yCYAAAA0yCYAAAA0yCYAAAA0yCYAAAA0yCYAAAA0yCYAAAA0yCYAAAA0yCYAAAA0yCYAAAA0yCYAAAA0yCYAAAA0yCYAAAA0yCYAAAA0yCYAAAA0yCYAAAA0zA0mDQ3N8vpdGrr1q2qra3VpEmTlJOToyeffFKXLl2SJJWUlGj8+PF67LHH5PF4jCwHAACYnKHB5A9/+IPi4uIkSWvWrFFOTo5eeeUVJSUlyePxqLGxUYWFhVq/fr02bdqkl156SWfPnjWyJAAAYGKGBZP33ntPR48e1UMPPSRJqqiokNPplCQ5nU6Vl5erqqpKKSkpstlsiomJUXp6urxer1ElAQAAk4syquPly5fr17/+tV577TVJUlNTk6xWqyTJ4XDI5/PJ7/crPj4+eI3dbpfP57tpnzU1NUaVC+BzMP5gpObmZv7GIMmgYPLaa69p4MCB+vrXvx5ss1gswdeBQOCan59u//R5n5WcnNzGlUrSMQP6BO4+xow/4BM1NTX8jUWYysrKG7YbEkx27dql//3vf9q1a5dOnjwpq9Wq2NhYNTc3KyYmRnV1dUpISFBiYqJ27doVvK6+vl4DBw40oiQAANAOGBJMVq1aFXy9du1aJSUl6e2331ZpaanGjRunsrIyZWRkKC0tTYsWLVJDQ4M6duwor9erhQsXGlESAABoBwx7xuSzZs2apXnz5qm4uFg9e/aU2+1WdHS08vLyNHXqVFksFuXm5spms92pkgAAgMkYHkxmzZoVfF1UVHTd710ul1wul9FlAACAdoCdXwEAgGkQTAAAgGkQTAAAgGkQTAAAgGmEFEyuXLlidB0AAAChBZNRo0ZpyZIlqqqqMroeAAAQwUIKJtu2bVNGRob+8pe/6Ic//KFWr16t9957z+jaAABAhAlpHxOr1aoRI0bogQce0L59+7RmzRq98cYb6tWrlxYsWKC+ffsaXScAAIgAIQWT/fv3a9u2bfJ6vbr//vuVn5+v/v376/jx48rLy9PWrVuNrhMAAESAkILJq6++qkceeUTPPPOMOnbsGGzv3bu3fvCDHxhWHAAAiCwhPWOSm5urqqqqYChZvHixjhw5IkmaOHGicdUBAICIEtKMSX5+vubMmRM8Hj9+vJ599llt3rzZsMIAwCj3zH8j3CXgho6FuwB8xvvLxt7x9wxpxqS1tVXp6enB4+985zsKBAKGFQUAACJTSDMmqampmj17tgYNGqQrV66ooqJCqampRtcGAAAiTEjBZOHChSovL9e///1vRUVFafr06dfMoAAAALSFkJZyTp48qSNHjujixYv6+OOPtX//fr3wwgtG1wYAACJMSDMmM2bMUEZGhnr06GF0PQAAIIKFFEzi4uKUl5dndC0AACDChRRM7rvvPv3pT3/S4MGDFRX1/5f06dPHsMIAAEDkCSmY7N27V5K0ffv2YJvFYtHGjRtvek1TU5Pmz5+vU6dO6eLFi5o5c6b69eunuXPnqrW1VQ6HQytWrJDValVJSYk2bNigDh06KDs7WxMmTPiStwUAANqjkILJpk2bJEmXL19WdHR0SB3/4x//0IABAzR9+nSdOHFCU6ZM0aBBg5STk6PRo0eroKBAHo9HbrdbhYWF8ng8io6OltvtVmZmpuLi4r74XQEAgHYppE/lVFRU6Pvf/76+973vSZKef/557dmz55bXjBkzRtOnT5ck1dbWKjExURUVFXI6nZIkp9Op8vJyVVVVKSUlRTabTTExMUpPT5fX6/0y9wQAANqpkGZM1qxZow0bNmj27NmSpMcff1wzZ87UAw888LnXTpw4USdPntSLL76oyZMny2q1SpIcDod8Pp/8fr/i4+OD59vtdvl8vhv2VVNTE0q5AAzA+AMiTzjGfUjBJCoqSt26dZPFYpEkde/ePfj687z66quqqanRr371q2uuubql/We3tg8EAjftOzk5OaT3vD18NwMQCmPGX7gw7oFQGDnuKysrb9ge0lJOr169tHr1ap05c0bbtm3TnDlzPvcTOdXV1aqtrZX0yY21trYqNjZWzc3NkqS6ujolJCQoMTFRfr8/eF19fb0cDkdINwUAAO4uIQWTxYsX6557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\n",
      "text/plain": [
       "<Figure size 648x216 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Survived: \n",
      " 0    549\n",
      "1    342\n",
      "Name: Survived, dtype: int64\n"
     ]
    },
    {
     "data": {
      "image/png": "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tEgVFRUaNWqUNm7ceNz4d911l37/+9/rqaeeUmJiop5//vlwHCYMxIoJet3w4cNlsVhkt9v13e9+V3379lVSUpKOHDmiTz/9VHl5ebJarWppaTmu3xtvvCG32y2PxyNJOnz4sDo6OvjXEmBCb7zxhvbv36+qqipJUnt7e/C1ESNGyGKxKDExUUOHDpUkJSQk6MCBA0pMTNT9998vv9+v5uZmXXXVVcF+Pp9PH374oWbNmiVJamtrU//+/XvxqNAbCCbodVFRUSd8vGvXLu3cuVMVFRWyWq266KKLjutntVo1c+ZM/ehHP+q1WgF8PVarVYsWLeoxjyWpb9++J3wcCAR09913a8aMGbr88sv1hz/8QW1tbceNmZycrIqKCmOLR1ixlQPTaGho0IABA2S1WvXSSy+pq6tLHR0dwdczMzP14osvSpL27dunBx98MFylAvgSx87X7du3q6ysLKR+n3zyic4991x1dHToH//4h44cORJ87eyzzw6OJ0kVFRV65513TnHlCDeCCUxj7Nix+vDDD3XDDTfoo48+0rhx47R48eLg65MmTdJZZ52lvLw8zZw5k68pAEzshhtu0M6dO5Wfn6/i4mJlZWWF3K+goECzZ8/WjTfeqGefffa4L3W9++67NX/+fOXn58vtduv888836hAQJtzHBAAAmAYrJgAAwDQIJgAAwDQIJgAAwDQIJgAAwDQIJgAAwDQIJgAAwDQIJgAAwDT+D7US8BaXzIuOAAAAAElFTkSuQmCC\n",
      "text/plain": [
       "<Figure size 648x216 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Sex: \n",
      " male      577\n",
      "female    314\n",
      "Name: Sex, dtype: int64\n"
     ]
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAiYAAADNCAYAAACM0rsuAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjEsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+j8jraAAAXN0lEQVR4nO3de1BU98HG8WcFtmjdisjFQmpfU21FRaxsbDUlabNGUJuK0VRKa1pvYyMhyQzxbquJaSPYtFHDxAyOl2CtjJvUIY0VTRs7JkFS2cjIlKkaTVotyq6XEoXVgPv+kXFfjZeufT3sj+z384+7P845+xz8IY/nnD1rCwQCAQEAABigS7gDAAAAXEYxAQAAxqCYAAAAY1BMAACAMSgmAADAGBQTAABgjOhwBwDw2fG1r31Nffr0UVRUlAKBgLp3764nn3xSI0aMuOE6NTU1Wrx4sXbt2tWBSQGYimIC4LYqLy9X7969JUm1tbV65JFHtGPHDsXHx4c5GYDOgFM5ACyTmZmpPn366L333pMk/f73v1d2drays7M1Z84cXbx48arlW1tb9cQTTyg7O1v33XefiouLg1/74x//qO9+97saM2aMHnjgAdXU1Nx0HEDnxBETAJZqa2uT3W7XsWPHVFJSom3btikpKUmFhYV6+eWXlZ6eHlz2d7/7nc6fP68dO3aoublZo0ePlsvlktPp1FNPPaVXXnlFqamp2rdvn3bt2qVvfOMbNxwH0DlRTABY5i9/+Yt8Pp+GDRumP/zhD/r617+u5ORkSdJzzz2nqKgo1dbWBpefNm2apkyZIpvNph49eqh///46duyYnE6nevXqpS1btigvL09Op1NOp1OSbjgOoHPiVA6A22rKlCnKyclRdna2Nm7cqLKyMn3+85/XmTNn9IUvfCG43Oc+9zlFR1/9f6MPPvhAhYWFGj16tHJyclRfX69Lly5Jkl588UX5fD49+OCDys3N1bvvvnvTcQCdE0dMANxWV178eqWePXsGrzWRpHPnzsnv91+1zNNPP61BgwaptLRUUVFRysvLC36tT58+evbZZ3Xp0iVt27ZNRUVF2rNnzw3HAXROHDEB0CHuvfdeeTweHTt2TIFAQEuWLJHb7b5qmVOnTiktLU1RUVF6++239eGHH+r8+fM6ffq0pk6dqnPnzqlLly7KyMiQzWa74TiAzosjJgA6RO/evfX000/rxz/+saKiopSenq6pU6dq//79wWUeeeQRPfPMM3rhhRd0//3369FHH9Wvf/1rDRw4UFlZWZo4caKioqIUExOjX/ziF4qPj7/uOIDOyxYIBALhDgEAACBxKgcAABiEYgIAAIxBMQEAAMagmAAAAGN0mnflXHl3SAAA0PllZmZeM9Zpiol0/R3A7dHQ0KC0tLRwxwCYizAGc9FaNzrgYFkxqa+v1+zZs/XlL39ZkvTVr35VM2bM0Ny5c9Xe3q7ExEStWLFCdrtdlZWV2rhxo7p06aLJkydr0qRJVsUCAAAGs6yYtLS0KDs7W4sWLQqOLViwQPn5+RozZoxKSkrkdruVm5ur0tJSud1uxcTEKDc3V6NGjVJcXJxV0QAAgKEsu/j1/Pnz14zV1NTI5XJJklwul6qrq1VXV6f09HQ5HA7FxsbK6XTK4/FYFQsAABjM0iMmtbW1mjFjhlpbW1VYWKjW1lbZ7XZJUmJiorxer3w+n+Lj44PrJSQkyOv1XnebDQ0NVsWNeH6/n+8vjMBchCmYi+FhWTEZMGCACgoK5HK5dPToUU2dOlVtbW3Br1++E/6n74gfCARu+CFcXIRkHS7ygimYizAFc9FaN7r41bJTOV/5yleCp2369u2rhIQENTc3Bz/m/OTJk0pKSlJycrJ8Pl9wvaamJiUmJloVCwAAGMyyIyZut1stLS16+OGH5fV6derUKT344IOqqqrS+PHjtXPnTmVlZSkjI0OLFy9Wc3OzoqKi5PF4tHDhQqtiAQBC8D/zXw93BEMcCXeAsPpg+bgOf03Lisn999+vJ598UlVVVbp48aKWLl2qtLQ0zZs3TxUVFUpJSVFubq5iYmJUVFSk6dOny2azqaCgQA6Hw6pYAADAYJYVkx49eqisrOya8fXr118zlpOTo5ycHKuiAACAToLPygEAAMagmAAAAGNQTAAAgDEoJgAAwBgUEwAAYAyKCQAAMAbFBAAAGINiAgAAjEExAQAAxqCYAAAAY1BMAACAMSgmAADAGBQTAABgDIoJAAAwBsUEAAAYg2ICAACMQTEBAADGoJgAAABjUEwAAIAxKCYAAMAYFBMAAGAMigkAADAGxQQAABjD0mLi9/vlcrn06quvqrGxUVOmTFF+fr4ef/xxXbx4UZJUWVmpiRMn6qGHHpLb7bYyDgAAMJylxeTFF19UXFycJGnVqlXKz8/X5s2blZqaKrfbrZaWFpWWlmrDhg0qLy/X2rVrdfbsWSsjAQAAg1lWTN5//30dPnxY3/72tyVJNTU1crlckiSXy6Xq6mrV1dUpPT1dDodDsbGxcjqd8ng8VkUCAACGi7Zqw8XFxfrZz36mbdu2SZJaW1tlt9slSYmJifJ6vfL5fIqPjw+uk5CQIK/Xe8NtNjQ0WBU34vn9fr6/MAJzETBHOH4WLSkm27Zt09ChQ/WlL30pOGaz2YKPA4HAVX9eOX7lcp+WlpZ2m5PisoaGBr6/MAJz0RRHwh0ABrDyZ7G2tva645YUk927d+uf//yndu/erRMnTshut6tr167y+/2KjY3VyZMnlZSUpOTkZO3evTu4XlNTk4YOHWpFJAAA0AlYUkyef/754OPVq1crNTVV7733nqqqqjR+/Hjt3LlTWVlZysjI0OLFi9Xc3KyoqCh5PB4tXLjQikgAAKATsOwak08rLCzUvHnzVFFRoZSUFOXm5iomJkZFRUWaPn26bDabCgoK5HA4OioSAAAwjOXFpLCwMPh4/fr113w9JydHOTk5VscAAACdAHd+BQAAxqCYAAAAY1BMAACAMSgmAADAGBQTAABgDIoJAAAwBsUEAAAYg2ICAACMQTEBAADGoJgAAABjUEwAAIAxKCYAAMAYFBMAAGAMigkAADAGxQQAABiDYgIAAIxBMQEAAMYIqZhcunTJ6hwAAAChFZPRo0frmWeeUV1dndV5AABABAupmGzfvl1ZWVl65ZVX9MMf/lArV67U+++/b3U2AAAQYaJDWchut+vee+/Vt771Lb3zzjtatWqVXn/9dd1xxx1asGCB+vfvb3VOAAAQAUIqJnv37tX27dvl8Xh09913a+nSpRo0aJCOHj2qoqIivfrqq1bnBAAAESCkYrJlyxZNmDBBS5YsUVRUVHC8b9+++v73v3/ddVpbWzV//nydOnVKFy5c0OzZszVgwADNnTtX7e3tSkxM1IoVK2S321VZWamNGzeqS5cumjx5siZNmnR79g4AAHQqIV1jUlBQoLq6umApWbZsmQ4dOiRJysvLu+46b775pgYPHqxNmzbp+eef1/Lly7Vq1Srl5+dr8+bNSk1NldvtVktLi0pLS7VhwwaVl5dr7dq1Onv27G3aPQAA0JmEVEyWLl2qkSNHBp9PnDhRTz311E3XGTt2rGbOnClJamxsVHJysmpqauRyuSRJLpdL1dXVqqurU3p6uhwOh2JjY+V0OuXxeP7b/QEAAJ1YSKdy2tvb5XQ6g88HDhyoQCAQ0gvk5eXpxIkTWrNmjaZOnSq73S5JSkxMlNfrlc/nU3x8fHD5hIQEeb3e626roaEhpNfErfP7/Xx/YQTmImCOcPwshlRMhgwZoscee0zDhg3TpUuXVFNToyFDhoT0Alu2bFFDQ4PmzJkjm80WHL9cbD5dcAKBwFXLXSktLS2k18Sta2ho4PsLIzAXTXEk3AFgACt/Fmtra687HtKpnIULF+oHP/iB2tra1KVLF82cOVPz5s276Tr19fVqbGyU9MmOtbe3q2vXrvL7/ZKkkydPKikpScnJyfL5fMH1mpqalJiYGNJOAQCAz5aQismJEyd06NAhXbhwQefOndPevXv1wgsv3HSdffv2ad26dZIkn8+nlpYWjRw5UlVVVZKknTt3KisrSxkZGTpw4ICam5t1/vx5eTyeq04bAQCAyBHSqZyf/vSnysrKUu/evUPecF5enhYtWqT8/Hz5/X79/Oc/1+DBgzVv3jxVVFQoJSVFubm5iomJUVFRkaZPny6bzaaCggI5HI7/eocAAEDnFVIxiYuLU1FR0S1tODY2Vs8999w14+vXr79mLCcnRzk5Obe0fQAA8NkTUjH55je/qd/+9rfKzMxUdPT/rdKvXz/LggEAgMgTUjF5++23JUk7duwIjtlsNr388svWpAIAABEppGJSXl4uSfr4448VExNjaSAAABC5QnpXTk1Njb73ve/pgQcekCT95je/0VtvvWVpMAAAEHlCKiarVq3Sxo0bg/cXefjhh7V69WpLgwEAgMgTUjGJjo5Wz549g3dk7dWr1w3vzgoAAPDfCukakzvuuEMrV67UmTNntH37du3atYt35AAAgNsupGKybNkyvfbaa8rMzNT+/fvlcrk0duxYq7MBAIAIE9KpnMrKSgUCAQ0dOlQDBw5UW1ubKisrrc4GAAAiTEhHTP7+978HH7e1tamurk79+/dXbm6uZcEAAEDkCamYfPqThNvb2/XYY49ZEggAAESukIpJa2vrVc+9Xq+OHDliSSAAABC5Qiom48aNCz622WxyOByaNm2aZaEAAEBkCqmY/PnPf7Y6BwAAQGjFxOVyXXc8EAjIZrPpT3/6020NBQAAIlNIxWT8+PHq16+fhg8frkuXLumvf/2rDh48qFmzZlmdDwAARJCQP8Rv7NixSkhIUFJSksaNGyePx6Nu3bqpW7duVmcEAAARIqQjJna7XSUlJRo6dKhsNpv279/PZ+UAAIDbLqQjJqtXr1ZqaqpqampUXV2tL37xiyotLbU6GwAAiDAhHTHp3r270tLSFBcXp3HjxqmpqUkOh8PqbAAAIMKEVEyKi4vV2Niof/zjHxo3bpwqKir073//W4sXL7Y6HwAAiCAhFZP6+nqVl5drypQpkqTCwkLl5+dbGqyj/M/818MdwSCRfTffD5aP+88LAQAsFdI1Jm1tbfr444+DF7yePn1aFy5csDQYAACIPCEdMZk2bZomT56sf/3rX5oxY4aOHDmihQsX/sf1SkpKVFtbq7a2Ns2aNUvp6emaO3eu2tvblZiYqBUrVshut6uyslIbN25Uly5dNHnyZE2aNOn/vWMAAKDzCamYpKamatOmTTp8+LBiYmLUt29fxcbG3nSdvXv36tChQ6qoqNCZM2c0YcIEjRgxQvn5+RozZoxKSkrkdruVm5ur0tJSud1uxcTEKDc3V6NGjVJcXNxt2UEAANB5hHQqZ/ny5bLb7RoyZIjS0tL+YymRpLvuuksrV66UJPXo0UOtra2qqakJ3t7e5XKpurpadXV1Sk9Pl8PhUGxsrJxOpzwez/9jlwAAQGcV0hGTbt26afTo0RowYIBiYmKC45eLx/VERUUF7wq7detW3XPPPXrrrbdkt9slSYmJifJ6vfL5fIqPjw+ul5CQIK/Xe91tNjQ0hBIX+K8wv8zg9/v5uwAMEY6fxZsWk2effVYLFizQtGnTJEkej0fDhg27pRd444035Ha7tW7dOmVnZwfHA4HAVX9eOX6ju8qmpaXd0muHJrLfiYL/Y838wq1qaGjg78II/NsIa/9drK2tve74TU/lXG5Kw4cP1/Dhw/XOO+8EHw8fPvw/vuiePXu0Zs0alZWVyeFwqGvXrvL7/ZKkkydPKikpScnJyfL5fMF1mpqalJiYGPKOAQCAz46bFpPrHc0I1UcffaSSkhK99NJLwQtZR44cqaqqKknSzp07lZWVpYyMDB04cEDNzc06f/68PB6PnE7nre4HAAD4DLjpqZxPn1K5lQ/u2759u86cOaMnnngiOLZ8+XItXrxYFRUVSklJUW5urmJiYlRUVKTp06fLZrOpoKCA290DABChbIGbHAYZNmyY7rzzTkmfHC05evSo7rzzzuB1IG63u8OC1tbWKjMz87Zvlzu/4rJw3/mVuYjLwj0XJeYjPmHlXLzR7/WbHjF57bXXLAsEAADwaTctJqmpqR2VAwAAILQbrAEAAHQEigkAADAGxQQAABiDYgIAAIxBMQEAAMagmAAAAGNQTAAAgDEoJgAAwBgUEwAAYAyKCQAAMAbFBAAAGINiAgAAjEExAQAAxqCYAAAAY1BMAACAMSgmAADAGBQTAABgDIoJAAAwBsUEAAAYg2ICAACMQTEBAADGsLSYHDx4UKNGjdKmTZskSY2NjZoyZYry8/P1+OOP6+LFi5KkyspKTZw4UQ899JDcbreVkQAAgMEsKyYtLS1atmyZRowYERxbtWqV8vPztXnzZqWmpsrtdqulpUWlpaXasGGDysvLtXbtWp09e9aqWAAAwGCWFRO73a6ysjIlJSUFx2pqauRyuSRJLpdL1dXVqqurU3p6uhwOh2JjY+V0OuXxeKyKBQAADBZt2YajoxUdffXmW1tbZbfbJUmJiYnyer3y+XyKj48PLpOQkCCv13vdbTY0NFgVF2B+wRjMRZgiHHPRsmJyPTabLfg4EAhc9eeV41cud6W0tDQLUh2xYJvojKyZX7eCuYhPhH8uSsxHSNbOxdra2uuOd+i7crp27Sq/3y9JOnnypJKSkpScnCyfzxdcpqmpSYmJiR0ZCwAAGKJDi8nIkSNVVVUlSdq5c6eysrKUkZGhAwcOqLm5WefPn5fH45HT6ezIWAAAwBCWncqpr69XcXGxjh8/rujoaFVVVelXv/qV5s+fr4qKCqWkpCg3N1cxMTEqKirS9OnTZbPZVFBQIIfDYVUsAABgMMuKyeDBg1VeXn7N+Pr1668Zy8nJUU5OjlVRAABAJ8GdXwEAgDEoJgAAwBgUEwAAYAyKCQAAMAbFBAAAGINiAgAAjEExAQAAxqCYAAAAY1BMAACAMSgmAADAGBQTAABgDIoJAAAwBsUEAAAYg2ICAACMQTEBAADGoJgAAABjUEwAAIAxKCYAAMAYFBMAAGAMigkAADAGxQQAABiDYgIAAIwRHe4Al/3yl79UXV2dbDabFi5cqCFDhoQ7EgAA6GBGFJN3331XH374oSoqKnT48GEtWLBAW7duDXcsAADQwYw4lVNdXa1Ro0ZJkvr166fm5madO3cuzKkAAEBHM+KIic/n06BBg4LPe/XqJa/Xq+7du1+1XG1t7W1/7Vce6n3bt4nOyYr5dSuYi7gs3HNRYj7iE+GYi0YUk0AgcM1zm8121VhmZmZHRgIAAGFgxKmc5ORk+Xy+4POmpiYlJCSEMREAAAgHI4rJ3XffraqqKknS3/72NyUlJV1zGgcAAHz2GXEqZ9iwYRo0aJDy8vJks9m0ZMmScEcCAABhYAt8+gIPRJyDBw9q9uzZ+slPfqIf/ehH4Y6DCFVSUqLa2lq1tbVp1qxZGj16dLgjIQK1trZq/vz5OnXqlC5cuKDZs2frO9/5TrhjRRQjjpggfFpaWrRs2TKNGDEi3FEQwfbu3atDhw6poqJCZ86c0YQJEygmCIs333xTgwcP1syZM3X8+HFNmzaNYtLBKCYRzm63q6ysTGVlZeGOggh21113Be/23KNHD7W2tqq9vV1RUVFhToZIM3bs2ODjxsZGJScnhzFNZKKYRLjo6GhFRzMNEF5RUVHq1q2bJGnr1q265557KCUIq7y8PJ04cUJr1qwJd5SIw28kAMZ444035Ha7tW7dunBHQYTbsmWLGhoaNGfOHFVWVl5zby1Yx4i3CwPAnj17tGbNGpWVlcnhcIQ7DiJUfX29GhsbJUlpaWlqb2/X6dOnw5wqslBMAITdRx99pJKSEr300kuKi4sLdxxEsH379gWP2Pl8PrW0tKhnz55hThVZeLtwhKuvr1dxcbGOHz+u6OhoJScna/Xq1fxyQIeqqKjQ6tWr1bdv3+BYcXGxUlJSwpgKkcjv92vRokVqbGyU3+/Xo48+qvvuuy/csSIKxQQAABiDUzkAAMAYFBMAAGAMigkAADAGxQQAABiDYgIAAIxBMQEAAMagmAAAAGP8LzF93B/Z4NyBAAAAAElFTkSuQmCC\n",
      "text/plain": [
       "<Figure size 648x216 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Pclass: \n",
      " 3    491\n",
      "1    216\n",
      "2    184\n",
      "Name: Pclass, dtype: int64\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 648x216 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Embarked: \n",
      " S    644\n",
      "C    168\n",
      "Q     77\n",
      "Name: Embarked, dtype: int64\n"
     ]
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAiYAAADNCAYAAACM0rsuAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjEsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+j8jraAAAcZ0lEQVR4nO3df1RUdf7H8dcoTKBNojBgWNuylkdKxJQ6aZFug4pROZUFh92sxDaPpHaW8rebHdsSaE+lebSjm6ltG+vYGp00qC3PsRbZZDZWNjY13bYMYVCTAkYD5/vHnuYr6dqEc5nL8Hz8w8zHez/zfs+55/g693PvHYvP5/MJAADABHqFugAAAIDvEEwAAIBpEEwAAIBpEEwAAIBpEEwAAIBpEEwAAIBpRIS6AADhp6amRsXFxaqvr5fP51NMTIweffRRHTt2TO+++66eeuop3XPPPZoyZYomT558xv6ff/65nnzySR04cECSFB0drYceekgZGRld3QqALkYwARBUPp9PM2bM0BNPPKFx48ZJksrLy5Wfn68dO3Zo/PjxPzhHQUGBJk+erNWrV0uSqqurde+992r79u26+OKLjSwfQIixlAMgqI4dOyaPx6PU1FT/2IQJE/T6669r+/btuu+++/zje/fu1ZQpUzRu3DgtXrxY7e3t/vHT909NTVVZWZkGDhyoyspK3XbbbVq+fLkmTpyorKwsffTRR13WHwBjEUwABFX//v2VkpKiqVOnavPmzfriiy8kSQMHDjxj28rKSm3atEnbt2/Xhx9+qPfee0+SdOONN2r27NnatGmTPv30U0lSQkKCLBaLJGn//v0aPny4ysrKdN9992np0qVd0xwAwxFMAASVxWLR+vXrNX78eG3cuFEOh0NZWVkqLy8/Y9uJEycqOjpa0dHRGjt2rP/MR3FxsX7xi1+otLRUt9xyi2666Sb98Y9/9O/Xp08fTZo0SdJ/z8bU1taqtbW1axoEYCiCCYCgs9lsmj17tt544w198MEHmjx5sn7961/L6/V22G7AgAEd9mlqapIkXXDBBcrLy9PmzZtVWVmpGTNmqLCwUDt37pQkXXTRRf6zJxdddJEk+fcF0L0RTAAE1eHDh7V7927/+7i4OP3qV7/SkCFDzggmx48f7/C6X79+Onr0qD744AP/+EUXXaS7775b6enp2rdvnyTpq6++OmOOmJgYQ/oB0LUIJgCCqq6uTvn5+aqpqfGP/eMf/9CXX36plpaWDtuWl5frxIkTamlp0c6dO5WWlqbW1lbNnj3bf3ZEkj777DNVV1dr1KhRkiSv16t33nlHklRWVqZhw4bpggsu6ILuABiN24UBBNXVV1+tZcuWaenSpfr666916tQpxcbG6plnnlFdXV2HbceMGaOpU6eqvr5e48aNU3p6unr16qXVq1drxYoVeuKJJ+Tz+dS3b18tWLBAqampqqys1KBBg1RVVaXi4mL17t1by5cvD1G3AILN4vP5fKEuAgACVVlZqcWLF+vtt98OdSkADMBSDgAAMA2CCQAAMA2WcgAAgGlwxgQAAJhGt7krp6qqKtQlAACAIPruEQCn6zbBRDp7A2ZUW1ur5OTkUJcRNPRjbvRjbuHWD8yrux1r/+uEA0s5AADANAgmAADANAgmAADANAgmAADANAgmAADANAy9K6e0tFTr1q1TRESE5syZoyFDhmju3Llqb2+X3W5XcXGxrFarSktLtWHDBvXq1UvZ2dmaMmWKkWUBAACTMiyYHDt2TKtWrdKWLVvU0tKilStX6q233lJubq4mTZqkoqIiuVwuOZ1OrVq1Si6XS5GRkXI6ncrIyFBMTIxRpXXw0/lvGjTzgaDO9u/lWUGdDwAAMzJsKaeiokKjR4/WhRdeqPj4eC1btkyVlZVyOBySJIfDoYqKClVXVyslJUU2m01RUVFKS0uT2+02qiwAAGBihp0x+eKLL+Tz+fTwww+roaFBs2bNUmtrq6xWqyTJbrfL4/GosbFRAwYM8O8XFxcnj8dz1jlra2uNKtf0Qtm71+sNq++efsyNfoDOCZdjzdBrTOrr6/X888/ryy+/1NSpU2WxWPz/9t1vB37/NwR9Pl+H7U5nzBPtgrvkYpRQPs2vuz1N8IfQj7nRD9A53e1Y6/Inv8bGxurqq69WRESEfvKTn6hv376Kjo6W1+uV9N/QEh8fr4SEBDU2Nvr3a2hokN1uN6osAABgYoYFkxtuuEG7du3SqVOndPToUbW0tGjMmDEqKyuTJJWXlys9PV2pqanas2ePmpqa1NzcLLfbrbS0NKPKAgAAJmbYUk5CQoImTpyoe++9V62trVq8eLFSUlI0b948lZSUKDExUU6nU5GRkSooKFBeXp4sFovy8/Nls9mMKgsAAJiYodeY5OTkKCcnp8PY+vXrz9guMzNTmZmZRpYCAAC6AZ78CgAATINgAgAATINgAgAATINgAgAATINgAgAATINgAgAATINgAgAATINgAgAATINgAgAATINgAgAATINgAgAATINgAgAATINgAgAATINgAgAATINgAgAATINgAgAATINgAgAATCPCqIlramo0c+ZMXXbZZZKkIUOGaPr06Zo7d67a29tlt9tVXFwsq9Wq0tJSbdiwQb169VJ2dramTJliVFkAAMDEDAsmLS0tmjhxohYtWuQfW7BggXJzczVp0iQVFRXJ5XLJ6XRq1apVcrlcioyMlNPpVEZGhmJiYowqDQAAmJRhSznNzc1njFVWVsrhcEiSHA6HKioqVF1drZSUFNlsNkVFRSktLU1ut9uosgAAgIkZesakqqpK06dPV2trq2bNmqXW1lZZrVZJkt1ul8fjUWNjowYMGODfLy4uTh6P56xz1tbWGlWu6YWyd6/XG1bfPf2YG/0AnRMux5phwWTo0KHKz8+Xw+HQwYMHdf/996utrc3/7z6fr8Pf08ctFstZ50xOTjag0gMGzBl8xvQemNra2pB+frDRj7nRD9A53e1Yq6qqOuu4YUs5gwcP9i/bJCUlKS4uTk1NTfJ6vZKk+vp6xcfHKyEhQY2Njf79GhoaZLfbjSoLAACYmGHBxOVyaePGjZIkj8ejI0eO6I477lBZWZkkqby8XOnp6UpNTdWePXvU1NSk5uZmud1upaWlGVUWAAAwMcOWcsaPH69HHnlEZWVlOnnypJYuXark5GTNmzdPJSUlSkxMlNPpVGRkpAoKCpSXlyeLxaL8/HzZbDajygIAACZmWDDp16+f1q5de8b4+vXrzxjLzMxUZmamUaUAAIBugie/AgAA0yCYAAAA0yCYAAAA0yCYAAAA0yCYAAAA0yCYAAAA0yCYAAAA0yCYAAAA0yCYAAAA0yCYAAAA0yCYAAAA0yCYAAAA0yCYAAAA0yCYAAAA0yCYAAAA0yCYAAAA0zA0mHi9XjkcDr322muqq6vTPffco9zcXM2ZM0cnT56UJJWWlurOO+/UXXfdJZfLZWQ5AADA5AwNJqtXr1ZMTIwkacWKFcrNzdUrr7yiQYMGyeVyqaWlRatWrdJLL72kTZs2ad26dfrqq6+MLAkAAJhYQMHk1KlTP3riTz/9VPv379e4ceMkSZWVlXI4HJIkh8OhiooKVVdXKyUlRTabTVFRUUpLS5Pb7f7RnwUAAMJDRCAbTZgwQePGjdOtt96q1NTUgCYuLCzUkiVLtHXrVklSa2urrFarJMlut8vj8aixsVEDBgzw7xMXFyePx/M/56ytrQ3os8NRKHv3er1h9d3Tj7nRD9A54XKsBRRMtm3bpoqKCm3ZskVFRUW69tprdcstt2jw4MFn3X7r1q0aMWKELr30Uv+YxWLxv/b5fB3+nj5++nbfl5ycHEi5P9IBA+YMPmN6D0xtbW1IPz/Y6Mfc6AfonO52rFVVVZ11PKBgYrVaNXbsWN1www3661//qhUrVujNN9/UJZdcogULFuiKK67osP2OHTv0+eefa8eOHTp8+LCsVquio6Pl9XoVFRWl+vp6xcfHKyEhQTt27PDv19DQoBEjRnS+SwAA0K0FFEx27dqlbdu2ye126/rrr9fSpUt11VVX6eDBgyooKNBrr73WYftnn33W/3rlypUaNGiQ/v73v6usrEyTJ09WeXm50tPTlZqaqsWLF6upqUm9e/eW2+3WwoULg9shAADoNgIKJq+++qpuv/12PfbYY+rdu7d/PCkpSXfffXdAHzRr1izNmzdPJSUlSkxMlNPpVGRkpAoKCpSXlyeLxaL8/HzZbLbOdQIAALq9gIJJfn6+tm/frrFjx0qSli1bppycHF1xxRXKyck5576zZs3yv16/fv0Z/56ZmanMzMwfUzMAAAhTAd0uvHTpUo0ZM8b//s4779Tjjz9uWFEAAKBnCiiYtLe3Ky0tzf/+yiuvPOOOGgAAgPMV0FLO8OHDNXv2bI0cOVKnTp1SZWWlhg8fbnRtAACghwkomCxcuFAVFRX65z//qYiICD3wwAMdzqAAAAAEQ0BLOYcPH9a+fft04sQJffPNN9q1a5eef/55o2sDAAA9TEBnTGbMmKH09HQNHDjQ6HoAAEAPFlAwiYmJUUFBgdG1AACAHi6gYHLdddfpD3/4g0aNGqWIiP/f5fLLLzesMAAA0PMEFEw++OADSdJbb73lH7NYLNq4caMxVQEAgB4poGCyadMmSdK3336ryMhIQwsCAAA9V0B35VRWVuq2227TrbfeKkl65pln9P777xtaGAAA6HkCCiYrVqzQhg0bZLfbJUlTp07VypUrDS0MAAD0PAEFk4iICPXv318Wi0WSFBsb638NAAAQLAFdY3LJJZfoueee07Fjx7Rt2za9/fbb3JEDAACCLqBgsmzZMr3xxhsaNWqUPvroIzkcDt18881G1wYAAHqYgJZySktL5fP5NGLECF155ZVqa2tTaWmp0bUBAIAeJqAzJp988on/dVtbm6qrq3XFFVfI6XQaVhgAAOh5Agom8+bN6/C+vb1ds2fPPuc+ra2tmj9/vo4cOaITJ05o5syZGjp0qObOnav29nbZ7XYVFxfLarWqtLRUGzZsUK9evZSdna0pU6Z0viMAANBtBRRMWltbO7z3eDw6cODAOfd57733NGzYMD3wwAM6dOiQpk2bppEjRyo3N1eTJk1SUVGRXC6XnE6nVq1aJZfLpcjISDmdTmVkZCgmJqbzXQEAgG4poGCSlZXlf22xWGSz2TRt2rRz7nP6xbF1dXVKSEhQZWWlHn/8cUmSw+HQSy+9pKSkJKWkpMhms0mS0tLS5Ha7ddNNN/3oZgAAQPcWUDB59913O/0BOTk5Onz4sNasWaP7779fVqtVkmS32+XxeNTY2KgBAwb4t4+Li5PH4znrXLW1tZ2uo7sLZe9erzesvnv6MTf6ATonXI61gIKJw+E467jP55PFYtFf/vKX/7nvq6++qtraWj366KMdHsrm8/k6/P3+nGeTnJwcSLk/0rmXpMzCmN4DU1tbG9LPDzb6MTf6ATqnux1rVVVVZx0PKJhMnjxZl19+ua699lqdOnVKH374ofbu3asHH3zwf+5TU1Oj2NhYXXzxxUpOTlZ7e7uio6Pl9XoVFRWl+vp6xcfHKyEhQTt27PDv19DQoBEjRvy47gAAQFgI+Ef8br75ZsXFxSk+Pl5ZWVlyu93q06eP+vTpc9Z9du/erRdffFGS1NjYqJaWFo0ZM0ZlZWWSpPLycqWnpys1NVV79uxRU1OTmpub5Xa7lZaWFqT2AABAdxLQGROr1aqioiKNGDFCFotFH3300Q/+Vk5OTo4WLVqk3Nxceb1e/eY3v9GwYcM0b948lZSUKDExUU6nU5GRkSooKFBeXp4sFovy8/P9F8ICAICeJaBgsnLlSr3++uuqrKyUz+fTz372M82YMeOc+0RFRel3v/vdGePr168/YywzM1OZmZkBlgwAAMJVQMHkwgsvVHJysmJiYpSVlaWGhgbOagAAgKALKJgUFhaqrq5O//nPf5SVlaWSkhIdP35cixcvNro+AADQgwR08WtNTY2effZZ9e3bV5I0a9Ysffzxx4YWBgAAep6AgklbW5u+/fZb/wWvR48e1YkTJwwtDAAA9DwBLeVMmzZN2dnZ+vLLLzV9+nQdOHBACxcuNLo2AADQwwQUTAYNGqSXX35Z+/fvV2RkpJKSkhQVFWV0bQAAoIcJaCln+fLlslqtGj58uJKTkwklAADAEAGdMenTp48mTJigoUOHKjIy0j/+3HPPGVYYAADoec4ZTJ566iktWLBA06ZNkyS53W6NHDmySwoDAAA9zzmDyXc/n3zttddKkp5//vkffOIrAABAZ53zGhOfz3fO9wAAAMF0zmDy/R/q+6Ef7gMAADgf51zKqamp0ZQpUyT992zJwYMHNWXKFPl8PlksFrlcri4pEgAA9AznDCZvvPFGV9UBAABw7mAyaNCgrqoDAAAgsAesAQAAdIWAHrDWWUVFRaqqqlJbW5sefPBBpaSkaO7cuWpvb5fdbldxcbGsVqtKS0u1YcMG9erVS9nZ2f7rWgAAQM9iWDDZtWuX9u3bp5KSEh07dky33367Ro8erdzcXE2aNElFRUVyuVxyOp1atWqVXC6XIiMj5XQ6lZGRoZiYGKNKC2s/nf+mQTMfCOps/16eFdT5AADhwbClnGuuucb/yPp+/fqptbVVlZWVcjgckiSHw6GKigpVV1crJSVFNptNUVFRSktLk9vtNqosAABgYoadMendu7f69OkjSdq8ebNuvPFGvf/++7JarZIku90uj8ejxsZGDRgwwL9fXFycPB7PWef87km0PVG49R7Kfrxeb1h9n/RjbuHWD8wrXI41Q68xkaR33nlHLpdLL774oiZOnOgf/+4psmd7uuz/epBbcnKyARUGd4nCKIH3Hm79BF9tbW1IPz/Y6Mfcwq0fmFd3O9aqqqrOOm7oXTk7d+7UmjVrtHbtWtlsNkVHR8vr9UqS6uvrFR8fr4SEBDU2Nvr3aWhokN1uN7IsAABgUoYFk6+//lpFRUV64YUX/BeyjhkzRmVlZZKk8vJypaenKzU1VXv27FFTU5Oam5vldruVlpZmVFkAAMDEDFvK2bZtm44dO6aHH37YP7Z8+XItXrxYJSUlSkxMlNPpVGRkpAoKCpSXlyeLxaL8/HzZbDajygIAACZmWDDJzs5Wdnb2GePr168/YywzM1OZmZlGlQIAALoJnvwKAABMg2ACAABMg2ACAABMg2ACAABMg2ACAABMg2ACAABMg2ACAABMg2ACAABMg2ACAABMg2ACAABMg2ACAABMg2ACAABMg2ACAABMg2ACAABMg2ACAABMg2ACAABMg2ACAABMw9BgsnfvXmVkZOjll1+WJNXV1emee+5Rbm6u5syZo5MnT0qSSktLdeedd+quu+6Sy+UysiQAAGBihgWTlpYWLVu2TKNHj/aPrVixQrm5uXrllVc0aNAguVwutbS0aNWqVXrppZe0adMmrVu3Tl999ZVRZQEAABMzLJhYrVatXbtW8fHx/rHKyko5HA5JksPhUEVFhaqrq5WSkiKbzaaoqCilpaXJ7XYbVRYAADCxCMMmjohQRETH6VtbW2W1WiVJdrtdHo9HjY2NGjBggH+buLg4eTyes85ZW1trVLmmF269h7Ifr9cbVt8n/ZhbuPUD8wqXY82wYHI2FovF/9rn83X4e/r46dudLjk52YCqDhgwZ/AF3nu49RN8tbW1If38YKMfcwu3fmBe3e1Yq6qqOut4l96VEx0dLa/XK0mqr69XfHy8EhIS1NjY6N+moaFBdru9K8sCAAAm0aXBZMyYMSorK5MklZeXKz09XampqdqzZ4+amprU3Nwst9uttLS0riwLAACYhGFLOTU1NSosLNShQ4cUERGhsrIyPf3005o/f75KSkqUmJgop9OpyMhIFRQUKC8vTxaLRfn5+bLZbEaVBQAATMywYDJs2DBt2rTpjPH169efMZaZmanMzEyjSgEAAN0ET34FAACmQTABAACmQTABAACmQTABAACmQTABAACmQTABAACmQTABAACmQTABAACmQTABAACmQTABAACmQTABAACmQTABAACmYdiP+AHB8NP5bxo084Ggzvbv5VlBnQ8AeirOmAAAANMgmAAAANNgKQfoQuG2NNUd+mGZDeheTBNMnnzySVVXV8tisWjhwoUaPnx4qEsCAABdzBTB5G9/+5s+++wzlZSUaP/+/VqwYIE2b94c6rIAAEAXM0UwqaioUEZGhiTp8ssvV1NTk7755htdeOGFIa4MQE/C0hS6Snc41qTQHG8Wn8/n6/JP/Z4lS5Zo7Nix/nCSm5ur3/72t0pKSvJvU1VVFaryAACAAUaNGnXGmCnOmHw/G/l8Plkslg5jZyseAACEF1PcLpyQkKDGxkb/+4aGBsXFxYWwIgAAEAqmCCbXX3+9ysrKJEkff/yx4uPjub4EAIAeyBRLOSNHjtRVV12lnJwcWSwWPfbYY6EuCQAAhIApLn4NJ+H2PJa9e/dq5syZuu+++/TLX/4y1OWct6KiIlVVVamtrU0PPvigJkyYEOqSOqW1tVXz58/XkSNHdOLECc2cOVM///nPQ13WefN6vcrKylJ+fr7uuOOOUJfTaTU1NZo5c6Yuu+wySdKQIUO0ZMmSEFeFcNXc3Kx58+bp+PHj+vbbb5Wfn6/09PRQl9VppjhjEi7C7XksLS0tWrZsmUaPHh3qUoJi165d2rdvn0pKSnTs2DHdfvvt3TaYvPfeexo2bJgeeOABHTp0SNOmTQuLYLJ69WrFxMSEuozz1tLSookTJ2rRokWhLgU9wJ///GclJSWpoKBA9fX1uvfee/XWW2+FuqxOI5gEUbg9j8VqtWrt2rVau3ZtqEsJimuuucZ/Bqtfv35qbW1Ve3u7evfuHeLKfrybb77Z/7qurk4JCQkhrCY4Pv30U+3fv1/jxo0LdSnnrbm5OdQloAfp37+/PvnkE0lSU1OT+vfvH+KKzg/BJIgaGxt11VVX+d/HxsbK4/F022ASERGhiIjwOUR69+6tPn36SJI2b96sG2+8sVuGktPl5OTo8OHDWrNmTahLOW+FhYVasmSJtm7dGupSzltLS4uqqqo0ffp0tba2atasWbruuutCXRbCVFZWll577TWNHz9eTU1NeuGFF0Jd0nkJn/91TCCQ57Eg9N555x25XC69+OKLoS7lvL366quqra3Vo48+qtLS0m57vG3dulUjRozQpZdeGupSgmLo0KHKz8+Xw+HQwYMHdf/996u8vFxWqzXUpSEMvf7660pMTNTvf/97/etf/9KiRYu0ZcuWUJfVaQSTIOJ5LOa3c+dOrVmzRuvWrZPNZgt1OZ1WU1Oj2NhYXXzxxUpOTlZ7e7uOHj2q2NjYUJfWKTt27NDnn3+uHTt26PDhw7JarRo4cKDGjBkT6tI6ZfDgwRo8eLAkKSkpSXFxcaqvrw+b4AVzcbvduuGGGyT9NxTX19erra2t257xNsVzTMIFz2Mxt6+//lpFRUV64YUXuv0Flrt37/af8WlsbFRLS0u3Xld+9tlntWXLFv3pT3/SXXfdpZkzZ3bbUCJJLpdLGzdulCR5PB4dOXIkLK4Dgjlddtllqq6uliQdOnRIffv27bahROJ24aB7+umntXv3bv/zWIYOHRrqkjqtpqZGhYWFOnTokCIiIpSQkKCVK1d22//US0pKtHLlyg6/wVRYWKjExMQQVtU5Xq9XixYtUl1dnbxerx566CHddNNNoS4rKFauXKlBgwZ169uFjx8/rkceeUQtLS06efKkHnroIY0dOzbUZSFMNTc3a+HChTpy5Ija2to0Z86cbn03JcEEAACYBks5AADANAgmAADANAgmAADANAgmAADANAgmAADANAgmAADANAgmAADANP4PPtJ7IrlVdCcAAAAASUVORK5CYII=\n",
      "text/plain": [
       "<Figure size 648x216 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "SibSp: \n",
      " 0    608\n",
      "1    209\n",
      "2     28\n",
      "4     18\n",
      "3     16\n",
      "8      7\n",
      "5      5\n",
      "Name: SibSp, dtype: int64\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 648x216 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Parch: \n",
      " 0    678\n",
      "1    118\n",
      "2     80\n",
      "5      5\n",
      "3      5\n",
      "4      4\n",
      "6      1\n",
      "Name: Parch, dtype: int64\n"
     ]
    }
   ],
   "source": [
    "category1 = [\"Survived\",\"Sex\",\"Pclass\",\"Embarked\",\"SibSp\",\"Parch\"]\n",
    "for c in category1:\n",
    "    bar_plot(c)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2020-09-08T17:54:25.813831Z",
     "iopub.status.busy": "2020-09-08T17:54:25.812786Z",
     "iopub.status.idle": "2020-09-08T17:54:25.817684Z",
     "shell.execute_reply": "2020-09-08T17:54:25.818569Z"
    },
    "papermill": {
     "duration": 0.054347,
     "end_time": "2020-09-08T17:54:25.818809",
     "exception": false,
     "start_time": "2020-09-08T17:54:25.764462",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<bound method IndexOpsMixin.value_counts of 0       NaN\n",
      "1       C85\n",
      "2       NaN\n",
      "3      C123\n",
      "4       NaN\n",
      "       ... \n",
      "886     NaN\n",
      "887     B42\n",
      "888     NaN\n",
      "889    C148\n",
      "890     NaN\n",
      "Name: Cabin, Length: 891, dtype: object> \n",
      "\n",
      "<bound method IndexOpsMixin.value_counts of 0                                Braund, Mr. Owen Harris\n",
      "1      Cumings, Mrs. John Bradley (Florence Briggs Th...\n",
      "2                                 Heikkinen, Miss. Laina\n",
      "3           Futrelle, Mrs. Jacques Heath (Lily May Peel)\n",
      "4                               Allen, Mr. William Henry\n",
      "                             ...                        \n",
      "886                                Montvila, Rev. Juozas\n",
      "887                         Graham, Miss. Margaret Edith\n",
      "888             Johnston, Miss. Catherine Helen \"Carrie\"\n",
      "889                                Behr, Mr. Karl Howell\n",
      "890                                  Dooley, Mr. Patrick\n",
      "Name: Name, Length: 891, dtype: object> \n",
      "\n",
      "<bound method IndexOpsMixin.value_counts of 0             A/5 21171\n",
      "1              PC 17599\n",
      "2      STON/O2. 3101282\n",
      "3                113803\n",
      "4                373450\n",
      "             ...       \n",
      "886              211536\n",
      "887              112053\n",
      "888          W./C. 6607\n",
      "889              111369\n",
      "890              370376\n",
      "Name: Ticket, Length: 891, dtype: object> \n",
      "\n"
     ]
    }
   ],
   "source": [
    "category2 = [\"Cabin\",\"Name\",\"Ticket\"]\n",
    "for c in category2:\n",
    "    print(\"{} \\n\".format(train_df[c].value_counts))"
   ]
  },
  {
   "cell_type": "markdown",
   "execution_count": null,
   "metadata": {
    "papermill": {
     "duration": 0.037615,
     "end_time": "2020-09-08T17:54:25.896273",
     "exception": false,
     "start_time": "2020-09-08T17:54:25.858658",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "Numerical Variable"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2020-09-08T17:54:25.980083Z",
     "iopub.status.busy": "2020-09-08T17:54:25.979123Z",
     "iopub.status.idle": "2020-09-08T17:54:25.982101Z",
     "shell.execute_reply": "2020-09-08T17:54:25.981457Z"
    },
    "papermill": {
     "duration": 0.047643,
     "end_time": "2020-09-08T17:54:25.982225",
     "exception": false,
     "start_time": "2020-09-08T17:54:25.934582",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "def plot_hist(variable):\n",
    "    plt.figure(figsize = (9,3))\n",
    "    plt.hist(train_df[variable], bins=50)\n",
    "    plt.xlabel(variable)\n",
    "    plt.ylabel(\"Frequency\")\n",
    "    plt.title(\"{} distribution with hist\".format(variable))\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2020-09-08T17:54:26.151370Z",
     "iopub.status.busy": "2020-09-08T17:54:26.149394Z",
     "iopub.status.idle": "2020-09-08T17:54:26.992807Z",
     "shell.execute_reply": "2020-09-08T17:54:26.992048Z"
    },
    "papermill": {
     "duration": 0.972533,
     "end_time": "2020-09-08T17:54:26.992928",
     "exception": false,
     "start_time": "2020-09-08T17:54:26.020395",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 648x216 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 648x216 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAikAAADbCAYAAACoVgElAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjEsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+j8jraAAAgAElEQVR4nO3deVxVdf7H8RdwxQ1zASHXIncZycSNXDAFSc1S00QERxutSceSsXJtxHAJM02trCGzXEZNNJVcoLIaKyS7lA2jpaaZYiKLiCIuwP394cP7kwH0utzLId7Px6PHg/M93/s9n3u/j4fn3fece66TxWKxICIiImIwzmVdgIiIiEhJFFJERETEkBRSRERExJAUUkRERMSQFFJERETEkBRSRERExJAUUkRuU4sWLQgKCuLhhx8mODiYxx9/nMTExLIu647auHEjI0eOLHHf5s2bCQ8Pv+EY1/Z78cUX2blz53X7b9u2jXPnzpW477XXXmPNmjXAlc//5MmTNzz+tQ4fPsyePXsA+OSTT5gyZcpNvf52XHu8a+tISkoiKCjohq8/fvw4rVu3LnHfqlWreP3116/7+r179/LTTz/dZNUiZcNU1gWI/BGsXLmSu+++GwCz2cwzzzzDjh07qFOnThlXZkzz5s27YZ/FixfTrl073Nzciu2bOHHibR3/008/JT8/nw4dOhAUFGRTOLhTrj3etXXcCWFhYTfss2HDBvz8/GjZsuUdOaaIPWklReQO8/Pzo3Hjxnz//fcArF+/nj59+tC7d2+GDx9OamoqAGlpafz5z3+mb9++BAYGsnDhwuu2A7z55psEBwfz0EMPMWvWLAoKCgAIDw9n+fLlDBs2jG7duvH3v/+dq89p3LhxI7169eLRRx9l48aNtGjRwqbxFi5cSJ8+fUhOTi7y/goLC3n55Zfp0aMHgwcPLvX/yq/XLzw8nM2bNwOwcOFCgoODCQ4OZsSIEaSlpTFlyhSOHDlCeHg43333HZMnT2bu3Ln079+f7du3M3nyZN566y3reB9//DH9+/enR48erF692vq+r139ubq9c+dO3nnnHVasWMErr7xSpF92djbPPfccwcHB9O3bl3/+85/W17do0YJNmzYxYMAAunbtyvvvv1/sPT///POsX78egIyMDFq0aMHXX38NQEpKCv379y+1jquW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YejQoYSEhPDjjz/aszwRERExMJO9Bj5//jxRUVH4+/sXaZs9ezatWrUq8TXffvstR48eZd26dRw6dIgpU6awfv16e5UoIiIiBma3lRRXV1diYmLw9PS0tuXm5l73NYmJiQQGBgLQtGlTcnJyOHfunL1KFBEREQOz20qKyWTCZCo6fG5uLm+88QY5OTl4eXkxffp0atWqZd2fkZGBj4+Pddvd3Z309HTc3NyKjLN//357lS0iIiIlKItzr91CSklCQkJo2rQp3t7eLF26lCVLlvDSSy9Z91ssliL9LRYLTk5OxcYp7XLR7Ttsp3FFRETKN/ude8FsNpfY7tBv9wQFBeHt7W39++effy6y38vLi4yMDOv2qVOn8PDwcGSJIiIiYhAODSl//etfOXHiBABJSUk0a9asyP4uXboQHx8PwL59+/D09Cx2qUdEREQqBrtd7klJSSE6OprU1FRMJhPx8fEMGzaM8ePHU61aNapWrcrcuXMBiIiIYO7cubRr1w4fHx9CQkJwcnJixowZ9ipPREREDM7J8r83ghic2WzGz8/PLmPfO3mrXcYVEREp7359pZ/dxi7t3K4nzoqIiIghKaSIiIiIISmkiIiIiCEppIiIiIghKaSIiIiIISmkiIiIiCEppIiIiIghKaSIiIiIISmkiIiIiCEppIiIiIghKaSIiIiIISmkiIiIiCHZNaQcOHCAwMBAVq1aBcDvv//OyJEjCQsLY+TIkaSnpxfpn5KSQvfu3QkPDyc8PJyoqCh7liciIiIGZrLXwOfPnycqKgp/f39r2+uvv84TTzxB3759Wb16NcuXL+fFF18s8prg4GCmTZtmr7JERESknLDbSoqrqysxMTF4enpa22bMmEFwcDAAtWvXJjs7u8hrcnNz7VWOiIiIlDN2W0kxmUyYTEWHr1atGgAFBQX861//Yty4cUX2nz9/HrPZzOjRo8nLy2P8+PF07ty52Nj79++3V9kiIiJSgrI499otpJSmoKCAF198kc6dOxe5FATQsmVLxo0bR69evThy5AijRo0iISEBV1fXIv1atWplp+oO22lcERGR8s1+514wm80ltjs8pEyZMoV77rmHv/3tb8X2NWnShCZNmgDg7e2Nh4cHaWlpNGrUyNFlioiISBlz6FeQt2zZQqVKlXj22WdL3B8bG8uKFSsASE9PJzMzEy8vL0eWKCIiIgZht5WUlJQUoqOjSU1NxWQyER8fT2ZmJpUrVyY8PBy4snISGRlJREQEc+fOJSgoiOeff574+HguXbpEZGRksUs9IiIiUjE4WSwWy406FRYW4uxsjOe+mc1m/Pz87DL2vZO32mVcERGR8u7XV/rZbezSzu02JY/evXsza9Ys9u7de8cLExERESmJTSFl27ZtdOvWjQ0bNjB8+HAWLVrEL7/8Yu/aREREpAKz6Z4UV1dXAgIC6Nq1K9988w2LFy9m69atNGzYkClTptCsWTN71ykiIiIVjE0hZffu3Wzbto3k5GS6dOlCZGQkPj4+HDlyhIkTJ7Jx40Z71ykiIiIVjE0hZe3atQwcOJAZM2bg4uJibff29uaJJ56wW3EiIiJScdl0T8q4cePYu3evNaBERUVx8OBBAEJCQuxXnYiIiFRYNoWUyMhIHnzwQev2448/zsyZM+1WlIiIiIhNIaWgoID27dtbt1u3bo0Nj1cRERERuWU23ZPi6+vLs88+S7t27SgsLCQpKQlfX1971yYiIiIVmE0hZerUqSQmJvLf//4Xk8nEmDFjiqysiIiIiNxpNl3uOXnyJAcPHuTixYucO3eO3bt388Ybb9i7NhEREanAbFpJ+etf/0q3bt24++677V2PiIiICGBjSKlVqxYTJ060dy0iIiIiVjZd7uncuTOrV6/mp59+4tChQ9b/buTAgQMEBgayatUqAH7//XfCw8MJDQ3lueee49KlS8VeM2fOHIYOHUpISAg//vjjTb4dERER+aOwaSXl66+/BmDHjh3WNicnJ1asWFHqa86fP09UVBT+/v7WtsWLFxMaGkqfPn2YN28esbGxhIaGWvd/++23HD16lHXr1nHo0CGmTJnC+vXrb/pNiYiISPlnU0hZuXIlAJcvX6ZSpUo2Dezq6kpMTAwxMTHWtqSkJOtD4Hr16sX7779fJKQkJiYSGBgIQNOmTcnJyeHcuXO4ubnZ9m5ERETkD8OmkJKUlMTs2bO5dOkSO3bsYOHChXTo0IGuXbuWPrDJhMlUdPi8vDxcXV0BqFu3Lunp6UX2Z2Rk4OPjY912d3cnPT29WEjZv3+/LWWLiIjIHVIW516bQsrixYv54IMPePbZZwEYMWIEY8eOvW5IKYmTk5P175KeWPu/bRaLpchrrmrVqtVNHdd2h+00roiISPlmv3MvmM3mEtttunHWZDJRu3Zta2Bwd3cvMTzcSNWqVblw4QIAaWlpeHp6Ftnv5eVFRkaGdfvUqVN4eHjc9HFERESk/LMppDRs2JBFixZx+vRptm3bRkREBE2bNr3pgz344IPEx8cDkJCQQLdu3Yrs79Kli3X/vn378PT01P0oIiIiFZRNl3uioqKIi4vDz8+PH374gV69etG3b9/rviYlJYXo6GhSU1MxmUzEx8czf/58Jk+ezLp166hfvz4DBgwAICIigrlz59KuXTt8fHwICQnBycmJGTNm3P47FBERkXLJyWLDzxlv2rSpxParIcORzGYzfn5+dhn73slb7TKuiIhIeffrK/3sNnZp53abVlJ+/vln69/5+fns3buXZs2alUlIERERkYrBppAyadKkItsFBQXWb/qIiIiI2INNISUvL6/Idnp6OocP6+u6IiIiYj82hZR+/f7/OpSTkxM1atTgySeftFtRIiIiIjaFlJ07d9q7DhEREZEibAopvXr1KrH96hNhP/vssztalIiIiIhNIeWxxx6jadOmdOzYkcLCQvbs2cOBAwd4+umn7V2fiIiIVFA2PXE2KSmJvn374uHhgaenJ/369SM5OZlq1apRrVo1e9coIiIiFZBNKymurq7MmzePtm3b4uTkxA8//HBLv90jIiIiYiubVlKWLFlCgwYNSEpKIjExkXr16vHmm2/auzYRERGpwGxaSXFzc6NVq1bUqlWLfv36cerUKWrUqGHv2kRERKQCsymkREdH8/vvv/Pbb7/Rr18/1q1bx5kzZ5g+fbq96xMREZEKyqbLPSkpKbz++utUr14dgPHjx7Nv3z67FiYiIiIVm00rKfn5+Vy+fNl6s2xWVhYXL1686YOtX7+eLVu2WLdTUlL4/vvvrdsDBgwochlp/vz5eHl53fRxREREpPyzKaQ8+eSTDB06lBMnTjB69GgOHz7M1KlTb/pgQ4YMYciQIQB8++23bN++vViflStX3vS4IiIi8sdjU0hp0KABq1at4tChQ1SqVAlvb2+qVKlyWwd+8803mT9/fpG23Nzc2xpTRERE/jhsCimvvPIK7733Hr6+vnfkoD/++CP16tWjbt26Rdqzs7OZOHEiqampdOrUiQkTJpT4PJb9+/ffkTpERETENmVx7rUppFSrVo3evXvTsmVLKlWqZG1ftGjRLR00NjaWgQMHFmuPiIjg0UcfpXLl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\n",
      "text/plain": [
       "<Figure size 648x216 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "numericVar = [\"Fare\",\"Age\",\"PassengerId\"]\n",
    "for n in numericVar:\n",
    "    plot_hist(n)"
   ]
  },
  {
   "cell_type": "raw",
   "execution_count": null,
   "metadata": {
    "papermill": {
     "duration": 0.038017,
     "end_time": "2020-09-08T17:54:27.117954",
     "exception": false,
     "start_time": "2020-09-08T17:54:27.079937",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "# Temel Veri Analizi\n",
    " "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {
    "_cell_guid": "79c7e3d0-c299-4dcb-8224-4455121ee9b0",
    "_uuid": "d629ff2d2480ee46fbb7e2d37f6b5fab8052498a",
    "execution": {
     "iopub.execute_input": "2020-09-08T17:54:27.211781Z",
     "iopub.status.busy": "2020-09-08T17:54:27.210823Z",
     "iopub.status.idle": "2020-09-08T17:54:27.214692Z",
     "shell.execute_reply": "2020-09-08T17:54:27.215261Z"
    },
    "papermill": {
     "duration": 0.058717,
     "end_time": "2020-09-08T17:54:27.215443",
     "exception": false,
     "start_time": "2020-09-08T17:54:27.156726",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Survived</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>0.629630</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>0.472826</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>0.242363</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Pclass  Survived\n",
       "0       1  0.629630\n",
       "1       2  0.472826\n",
       "2       3  0.242363"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Pclass vs Survived\n",
    "train_df[[\"Pclass\",\"Survived\"]].groupby([\"Pclass\"], as_index = False).mean().sort_values(by = \"Survived\", ascending = False )\n",
    "#bu kod ile grupla,sırala ve ortalamasını göster diyoruz\n",
    "#pclassa göre grupladık,surviveda göre azalan şekilde sıraladık"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2020-09-08T17:54:27.309482Z",
     "iopub.status.busy": "2020-09-08T17:54:27.301220Z",
     "iopub.status.idle": "2020-09-08T17:54:27.314733Z",
     "shell.execute_reply": "2020-09-08T17:54:27.314006Z"
    },
    "papermill": {
     "duration": 0.060651,
     "end_time": "2020-09-08T17:54:27.314860",
     "exception": false,
     "start_time": "2020-09-08T17:54:27.254209",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Sex</th>\n",
       "      <th>Survived</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>female</td>\n",
       "      <td>0.742038</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>male</td>\n",
       "      <td>0.188908</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "      Sex  Survived\n",
       "0  female  0.742038\n",
       "1    male  0.188908"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Sex vs Survived\n",
    "train_df[[\"Sex\",\"Survived\"]].groupby([\"Sex\"], as_index = False).mean().sort_values(by = \"Survived\", ascending = False )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2020-09-08T17:54:27.409654Z",
     "iopub.status.busy": "2020-09-08T17:54:27.408705Z",
     "iopub.status.idle": "2020-09-08T17:54:27.413232Z",
     "shell.execute_reply": "2020-09-08T17:54:27.412641Z"
    },
    "papermill": {
     "duration": 0.059307,
     "end_time": "2020-09-08T17:54:27.413361",
     "exception": false,
     "start_time": "2020-09-08T17:54:27.354054",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Survived</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>0.535885</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2</td>\n",
       "      <td>0.464286</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>0.345395</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>3</td>\n",
       "      <td>0.250000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>4</td>\n",
       "      <td>0.166667</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>5</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>8</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   SibSp  Survived\n",
       "1      1  0.535885\n",
       "2      2  0.464286\n",
       "0      0  0.345395\n",
       "3      3  0.250000\n",
       "4      4  0.166667\n",
       "5      5  0.000000\n",
       "6      8  0.000000"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#SibSp vs Survived\n",
    "train_df[[\"SibSp\",\"Survived\"]].groupby([\"SibSp\"], as_index = False).mean().sort_values(by = \"Survived\", ascending = False )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2020-09-08T17:54:27.501424Z",
     "iopub.status.busy": "2020-09-08T17:54:27.499978Z",
     "iopub.status.idle": "2020-09-08T17:54:27.512019Z",
     "shell.execute_reply": "2020-09-08T17:54:27.511399Z"
    },
    "papermill": {
     "duration": 0.059505,
     "end_time": "2020-09-08T17:54:27.512142",
     "exception": false,
     "start_time": "2020-09-08T17:54:27.452637",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Parch</th>\n",
       "      <th>Survived</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>3</td>\n",
       "      <td>0.600000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>0.550847</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2</td>\n",
       "      <td>0.500000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>0.343658</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>5</td>\n",
       "      <td>0.200000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>4</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>6</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Parch  Survived\n",
       "3      3  0.600000\n",
       "1      1  0.550847\n",
       "2      2  0.500000\n",
       "0      0  0.343658\n",
       "5      5  0.200000\n",
       "4      4  0.000000\n",
       "6      6  0.000000"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Parch vs Survived\n",
    "train_df[[\"Parch\",\"Survived\"]].groupby([\"Parch\"], as_index = False).mean().sort_values(by = \"Survived\", ascending = False )"
   ]
  },
  {
   "cell_type": "markdown",
   "execution_count": null,
   "metadata": {
    "papermill": {
     "duration": 0.039572,
     "end_time": "2020-09-08T17:54:27.591442",
     "exception": false,
     "start_time": "2020-09-08T17:54:27.551870",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "Outlier Detection"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2020-09-08T17:54:27.682342Z",
     "iopub.status.busy": "2020-09-08T17:54:27.681556Z",
     "iopub.status.idle": "2020-09-08T17:54:27.685141Z",
     "shell.execute_reply": "2020-09-08T17:54:27.684422Z"
    },
    "papermill": {
     "duration": 0.053703,
     "end_time": "2020-09-08T17:54:27.685267",
     "exception": false,
     "start_time": "2020-09-08T17:54:27.631564",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "def detect_outliers(df,features):\n",
    "    outlier_indices = []\n",
    "    \n",
    "    for c in features:\n",
    "        #1st quartile\n",
    "        Q1 = np.percentile(df[c],25)\n",
    "        #3rd quartile\n",
    "        Q3 = np.percentile(df[c],75)\n",
    "        #IQR\n",
    "        IQR = Q3 - Q1\n",
    "        #Outlier step\n",
    "        outlier_step = IQR * 1.5\n",
    "        #detect outlier and their indeces\n",
    "        outlier_list_col = df[(df[c] < Q1 - outlier_step) | (df[c] > Q1 + outlier_step)].index\n",
    "        #store indeces:bulduğumuz indeksleri depolıycaz ki sonradasn çıkarabilelim\n",
    "        outlier_indices.extend(outlier_list_col)\n",
    "        \n",
    "    outlier_indices = Counter(outlier_indices)\n",
    "    multiple_outliers = list(i for i, v in outlier_indices.items() if v > 2)\n",
    "    \n",
    "    return multiple_outliers"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2020-09-08T17:54:27.771900Z",
     "iopub.status.busy": "2020-09-08T17:54:27.771152Z",
     "iopub.status.idle": "2020-09-08T17:54:27.820074Z",
     "shell.execute_reply": "2020-09-08T17:54:27.819338Z"
    },
    "papermill": {
     "duration": 0.095106,
     "end_time": "2020-09-08T17:54:27.820196",
     "exception": false,
     "start_time": "2020-09-08T17:54:27.725090",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Name</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>Embarked</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>27</th>\n",
       "      <td>28</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>Fortune, Mr. Charles Alexander</td>\n",
       "      <td>male</td>\n",
       "      <td>19.0</td>\n",
       "      <td>3</td>\n",
       "      <td>2</td>\n",
       "      <td>19950</td>\n",
       "      <td>263.000</td>\n",
       "      <td>C23 C25 C27</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>59</th>\n",
       "      <td>60</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Goodwin, Master. William Frederick</td>\n",
       "      <td>male</td>\n",
       "      <td>11.0</td>\n",
       "      <td>5</td>\n",
       "      <td>2</td>\n",
       "      <td>CA 2144</td>\n",
       "      <td>46.900</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>71</th>\n",
       "      <td>72</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Goodwin, Miss. Lillian Amy</td>\n",
       "      <td>female</td>\n",
       "      <td>16.0</td>\n",
       "      <td>5</td>\n",
       "      <td>2</td>\n",
       "      <td>CA 2144</td>\n",
       "      <td>46.900</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>88</th>\n",
       "      <td>89</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Fortune, Miss. Mabel Helen</td>\n",
       "      <td>female</td>\n",
       "      <td>23.0</td>\n",
       "      <td>3</td>\n",
       "      <td>2</td>\n",
       "      <td>19950</td>\n",
       "      <td>263.000</td>\n",
       "      <td>C23 C25 C27</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>159</th>\n",
       "      <td>160</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Sage, Master. Thomas Henry</td>\n",
       "      <td>male</td>\n",
       "      <td>NaN</td>\n",
       "      <td>8</td>\n",
       "      <td>2</td>\n",
       "      <td>CA. 2343</td>\n",
       "      <td>69.550</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>180</th>\n",
       "      <td>181</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Sage, Miss. Constance Gladys</td>\n",
       "      <td>female</td>\n",
       "      <td>NaN</td>\n",
       "      <td>8</td>\n",
       "      <td>2</td>\n",
       "      <td>CA. 2343</td>\n",
       "      <td>69.550</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>201</th>\n",
       "      <td>202</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Sage, Mr. Frederick</td>\n",
       "      <td>male</td>\n",
       "      <td>NaN</td>\n",
       "      <td>8</td>\n",
       "      <td>2</td>\n",
       "      <td>CA. 2343</td>\n",
       "      <td>69.550</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>311</th>\n",
       "      <td>312</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Ryerson, Miss. Emily Borie</td>\n",
       "      <td>female</td>\n",
       "      <td>18.0</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>PC 17608</td>\n",
       "      <td>262.375</td>\n",
       "      <td>B57 B59 B63 B66</td>\n",
       "      <td>C</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>324</th>\n",
       "      <td>325</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Sage, Mr. George John Jr</td>\n",
       "      <td>male</td>\n",
       "      <td>NaN</td>\n",
       "      <td>8</td>\n",
       "      <td>2</td>\n",
       "      <td>CA. 2343</td>\n",
       "      <td>69.550</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>341</th>\n",
       "      <td>342</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Fortune, Miss. Alice Elizabeth</td>\n",
       "      <td>female</td>\n",
       "      <td>24.0</td>\n",
       "      <td>3</td>\n",
       "      <td>2</td>\n",
       "      <td>19950</td>\n",
       "      <td>263.000</td>\n",
       "      <td>C23 C25 C27</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>386</th>\n",
       "      <td>387</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Goodwin, Master. Sidney Leonard</td>\n",
       "      <td>male</td>\n",
       "      <td>1.0</td>\n",
       "      <td>5</td>\n",
       "      <td>2</td>\n",
       "      <td>CA 2144</td>\n",
       "      <td>46.900</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>480</th>\n",
       "      <td>481</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Goodwin, Master. Harold Victor</td>\n",
       "      <td>male</td>\n",
       "      <td>9.0</td>\n",
       "      <td>5</td>\n",
       "      <td>2</td>\n",
       "      <td>CA 2144</td>\n",
       "      <td>46.900</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>683</th>\n",
       "      <td>684</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Goodwin, Mr. Charles Edward</td>\n",
       "      <td>male</td>\n",
       "      <td>14.0</td>\n",
       "      <td>5</td>\n",
       "      <td>2</td>\n",
       "      <td>CA 2144</td>\n",
       "      <td>46.900</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>742</th>\n",
       "      <td>743</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Ryerson, Miss. Susan Parker \"Suzette\"</td>\n",
       "      <td>female</td>\n",
       "      <td>21.0</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>PC 17608</td>\n",
       "      <td>262.375</td>\n",
       "      <td>B57 B59 B63 B66</td>\n",
       "      <td>C</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>792</th>\n",
       "      <td>793</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Sage, Miss. Stella Anna</td>\n",
       "      <td>female</td>\n",
       "      <td>NaN</td>\n",
       "      <td>8</td>\n",
       "      <td>2</td>\n",
       "      <td>CA. 2343</td>\n",
       "      <td>69.550</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>846</th>\n",
       "      <td>847</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Sage, Mr. Douglas Bullen</td>\n",
       "      <td>male</td>\n",
       "      <td>NaN</td>\n",
       "      <td>8</td>\n",
       "      <td>2</td>\n",
       "      <td>CA. 2343</td>\n",
       "      <td>69.550</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>863</th>\n",
       "      <td>864</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Sage, Miss. Dorothy Edith \"Dolly\"</td>\n",
       "      <td>female</td>\n",
       "      <td>NaN</td>\n",
       "      <td>8</td>\n",
       "      <td>2</td>\n",
       "      <td>CA. 2343</td>\n",
       "      <td>69.550</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "     PassengerId  Survived  Pclass                                   Name  \\\n",
       "27            28         0       1         Fortune, Mr. Charles Alexander   \n",
       "59            60         0       3     Goodwin, Master. William Frederick   \n",
       "71            72         0       3             Goodwin, Miss. Lillian Amy   \n",
       "88            89         1       1             Fortune, Miss. Mabel Helen   \n",
       "159          160         0       3             Sage, Master. Thomas Henry   \n",
       "180          181         0       3           Sage, Miss. Constance Gladys   \n",
       "201          202         0       3                    Sage, Mr. Frederick   \n",
       "311          312         1       1             Ryerson, Miss. Emily Borie   \n",
       "324          325         0       3               Sage, Mr. George John Jr   \n",
       "341          342         1       1         Fortune, Miss. Alice Elizabeth   \n",
       "386          387         0       3        Goodwin, Master. Sidney Leonard   \n",
       "480          481         0       3         Goodwin, Master. Harold Victor   \n",
       "683          684         0       3            Goodwin, Mr. Charles Edward   \n",
       "742          743         1       1  Ryerson, Miss. Susan Parker \"Suzette\"   \n",
       "792          793         0       3                Sage, Miss. Stella Anna   \n",
       "846          847         0       3               Sage, Mr. Douglas Bullen   \n",
       "863          864         0       3      Sage, Miss. Dorothy Edith \"Dolly\"   \n",
       "\n",
       "        Sex   Age  SibSp  Parch    Ticket     Fare            Cabin Embarked  \n",
       "27     male  19.0      3      2     19950  263.000      C23 C25 C27        S  \n",
       "59     male  11.0      5      2   CA 2144   46.900              NaN        S  \n",
       "71   female  16.0      5      2   CA 2144   46.900              NaN        S  \n",
       "88   female  23.0      3      2     19950  263.000      C23 C25 C27        S  \n",
       "159    male   NaN      8      2  CA. 2343   69.550              NaN        S  \n",
       "180  female   NaN      8      2  CA. 2343   69.550              NaN        S  \n",
       "201    male   NaN      8      2  CA. 2343   69.550              NaN        S  \n",
       "311  female  18.0      2      2  PC 17608  262.375  B57 B59 B63 B66        C  \n",
       "324    male   NaN      8      2  CA. 2343   69.550              NaN        S  \n",
       "341  female  24.0      3      2     19950  263.000      C23 C25 C27        S  \n",
       "386    male   1.0      5      2   CA 2144   46.900              NaN        S  \n",
       "480    male   9.0      5      2   CA 2144   46.900              NaN        S  \n",
       "683    male  14.0      5      2   CA 2144   46.900              NaN        S  \n",
       "742  female  21.0      2      2  PC 17608  262.375  B57 B59 B63 B66        C  \n",
       "792  female   NaN      8      2  CA. 2343   69.550              NaN        S  \n",
       "846    male   NaN      8      2  CA. 2343   69.550              NaN        S  \n",
       "863  female   NaN      8      2  CA. 2343   69.550              NaN        S  "
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_df.loc[detect_outliers(train_df,[\"Age\",\"SibSp\",\"Parch\",\"Fare\"])]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2020-09-08T17:54:27.909194Z",
     "iopub.status.busy": "2020-09-08T17:54:27.906884Z",
     "iopub.status.idle": "2020-09-08T17:54:27.918641Z",
     "shell.execute_reply": "2020-09-08T17:54:27.918015Z"
    },
    "papermill": {
     "duration": 0.058381,
     "end_time": "2020-09-08T17:54:27.918761",
     "exception": false,
     "start_time": "2020-09-08T17:54:27.860380",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# drop outliers\n",
    "train_df = train_df.drop(detect_outliers(train_df,[\"Age\",\"SibSp\",\"Parch\",\"Fare\"]),axis = 0).reset_index(drop = True)"
   ]
  },
  {
   "cell_type": "markdown",
   "execution_count": null,
   "metadata": {
    "papermill": {
     "duration": 0.040049,
     "end_time": "2020-09-08T17:54:27.999426",
     "exception": false,
     "start_time": "2020-09-08T17:54:27.959377",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "\n",
    "Missing Value\n",
    " Find Missing Value\n",
    " Fill Missing Value"
   ]
  },
  {
   "cell_type": "markdown",
   "execution_count": null,
   "metadata": {
    "papermill": {
     "duration": 0.039858,
     "end_time": "2020-09-08T17:54:28.079816",
     "exception": false,
     "start_time": "2020-09-08T17:54:28.039958",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "#missing valuları bulmadan önce train ve test dataframelerini birleştireceğiz.\n",
    "#train dataframein uzunluğunu kaybetmemek için len metodu kullanırız"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2020-09-08T17:54:28.172140Z",
     "iopub.status.busy": "2020-09-08T17:54:28.171335Z",
     "iopub.status.idle": "2020-09-08T17:54:28.174824Z",
     "shell.execute_reply": "2020-09-08T17:54:28.175346Z"
    },
    "papermill": {
     "duration": 0.055257,
     "end_time": "2020-09-08T17:54:28.175512",
     "exception": false,
     "start_time": "2020-09-08T17:54:28.120255",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "train_df_len = len(train_df)\n",
    "train_df = pd.concat([train_df,test_df],axis = 0).reset_index(drop = True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2020-09-08T17:54:28.275873Z",
     "iopub.status.busy": "2020-09-08T17:54:28.274903Z",
     "iopub.status.idle": "2020-09-08T17:54:28.278933Z",
     "shell.execute_reply": "2020-09-08T17:54:28.279494Z"
    },
    "papermill": {
     "duration": 0.06371,
     "end_time": "2020-09-08T17:54:28.279700",
     "exception": false,
     "start_time": "2020-09-08T17:54:28.215990",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Name</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>Embarked</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>0.0</td>\n",
       "      <td>3</td>\n",
       "      <td>Braund, Mr. Owen Harris</td>\n",
       "      <td>male</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>A/5 21171</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1</td>\n",
       "      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>\n",
       "      <td>female</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>PC 17599</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>C85</td>\n",
       "      <td>C</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>1.0</td>\n",
       "      <td>3</td>\n",
       "      <td>Heikkinen, Miss. Laina</td>\n",
       "      <td>female</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>STON/O2. 3101282</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1</td>\n",
       "      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>\n",
       "      <td>female</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>113803</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>C123</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>0.0</td>\n",
       "      <td>3</td>\n",
       "      <td>Allen, Mr. William Henry</td>\n",
       "      <td>male</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>373450</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   PassengerId  Survived  Pclass  \\\n",
       "0            1       0.0       3   \n",
       "1            2       1.0       1   \n",
       "2            3       1.0       3   \n",
       "3            4       1.0       1   \n",
       "4            5       0.0       3   \n",
       "\n",
       "                                                Name     Sex   Age  SibSp  \\\n",
       "0                            Braund, Mr. Owen Harris    male  22.0      1   \n",
       "1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   \n",
       "2                             Heikkinen, Miss. Laina  female  26.0      0   \n",
       "3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   \n",
       "4                           Allen, Mr. William Henry    male  35.0      0   \n",
       "\n",
       "   Parch            Ticket     Fare Cabin Embarked  \n",
       "0      0         A/5 21171   7.2500   NaN        S  \n",
       "1      0          PC 17599  71.2833   C85        C  \n",
       "2      0  STON/O2. 3101282   7.9250   NaN        S  \n",
       "3      0            113803  53.1000  C123        S  \n",
       "4      0            373450   8.0500   NaN        S  "
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "execution_count": null,
   "metadata": {
    "papermill": {
     "duration": 0.04049,
     "end_time": "2020-09-08T17:54:28.360848",
     "exception": false,
     "start_time": "2020-09-08T17:54:28.320358",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "Find Missing Value\n"
   ]
  },
  {
   "cell_type": "markdown",
   "execution_count": null,
   "metadata": {
    "papermill": {
     "duration": 0.040377,
     "end_time": "2020-09-08T17:54:28.441884",
     "exception": false,
     "start_time": "2020-09-08T17:54:28.401507",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "#bu kod ile train dataframein içinde missing value var mı baktık,\n",
    "#hangi columnlarda olduğunu tespit ettim"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2020-09-08T17:54:28.531825Z",
     "iopub.status.busy": "2020-09-08T17:54:28.529864Z",
     "iopub.status.idle": "2020-09-08T17:54:28.536755Z",
     "shell.execute_reply": "2020-09-08T17:54:28.536000Z"
    },
    "papermill": {
     "duration": 0.053987,
     "end_time": "2020-09-08T17:54:28.536885",
     "exception": false,
     "start_time": "2020-09-08T17:54:28.482898",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['Survived', 'Age', 'Fare', 'Cabin', 'Embarked'], dtype='object')"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_df.columns[train_df.isnull().any()]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2020-09-08T17:54:28.630857Z",
     "iopub.status.busy": "2020-09-08T17:54:28.629704Z",
     "iopub.status.idle": "2020-09-08T17:54:28.633764Z",
     "shell.execute_reply": "2020-09-08T17:54:28.634631Z"
    },
    "papermill": {
     "duration": 0.056447,
     "end_time": "2020-09-08T17:54:28.634858",
     "exception": false,
     "start_time": "2020-09-08T17:54:28.578411",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "PassengerId       0\n",
       "Survived        418\n",
       "Pclass            0\n",
       "Name              0\n",
       "Sex               0\n",
       "Age             256\n",
       "SibSp             0\n",
       "Parch             0\n",
       "Ticket            0\n",
       "Fare              1\n",
       "Cabin          1002\n",
       "Embarked          2\n",
       "dtype: int64"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_df.isnull().sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "execution_count": null,
   "metadata": {
    "papermill": {
     "duration": 0.042699,
     "end_time": "2020-09-08T17:54:28.724904",
     "exception": false,
     "start_time": "2020-09-08T17:54:28.682205",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "#kaç tane olduğunu öğreniriz\n",
    "#256 tane yolcumuun yaşını bilmiyorum"
   ]
  },
  {
   "cell_type": "markdown",
   "execution_count": null,
   "metadata": {
    "papermill": {
     "duration": 0.041656,
     "end_time": "2020-09-08T17:54:28.811827",
     "exception": false,
     "start_time": "2020-09-08T17:54:28.770171",
     "status": "completed"
    },
    "tags": []
   },
   "source": []
  },
  {
   "cell_type": "markdown",
   "execution_count": null,
   "metadata": {
    "papermill": {
     "duration": 0.040726,
     "end_time": "2020-09-08T17:54:28.893847",
     "exception": false,
     "start_time": "2020-09-08T17:54:28.853121",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "Fill Missing Value\n",
    "Embarked has 2 missing value\n",
    "Fare has only 1\n",
    "embarkes ve fare ı doldurucaz"
   ]
  },
  {
   "cell_type": "markdown",
   "execution_count": null,
   "metadata": {
    "papermill": {
     "duration": 0.041301,
     "end_time": "2020-09-08T17:54:28.977198",
     "exception": false,
     "start_time": "2020-09-08T17:54:28.935897",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "#Veriyi kaybetmektense doldurmak daha mantıklı\n",
    "#embarkedın nerde boş olduğunu bakalım"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2020-09-08T17:54:29.072172Z",
     "iopub.status.busy": "2020-09-08T17:54:29.071393Z",
     "iopub.status.idle": "2020-09-08T17:54:29.089167Z",
     "shell.execute_reply": "2020-09-08T17:54:29.088554Z"
    },
    "papermill": {
     "duration": 0.06573,
     "end_time": "2020-09-08T17:54:29.089307",
     "exception": false,
     "start_time": "2020-09-08T17:54:29.023577",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Name</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>Embarked</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>59</th>\n",
       "      <td>62</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1</td>\n",
       "      <td>Icard, Miss. Amelie</td>\n",
       "      <td>female</td>\n",
       "      <td>38.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>113572</td>\n",
       "      <td>80.0</td>\n",
       "      <td>B28</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>814</th>\n",
       "      <td>830</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1</td>\n",
       "      <td>Stone, Mrs. George Nelson (Martha Evelyn)</td>\n",
       "      <td>female</td>\n",
       "      <td>62.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>113572</td>\n",
       "      <td>80.0</td>\n",
       "      <td>B28</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "     PassengerId  Survived  Pclass                                       Name  \\\n",
       "59            62       1.0       1                        Icard, Miss. Amelie   \n",
       "814          830       1.0       1  Stone, Mrs. George Nelson (Martha Evelyn)   \n",
       "\n",
       "        Sex   Age  SibSp  Parch  Ticket  Fare Cabin Embarked  \n",
       "59   female  38.0      0      0  113572  80.0   B28      NaN  \n",
       "814  female  62.0      0      0  113572  80.0   B28      NaN  "
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_df[train_df[\"Embarked\"].isnull()]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2020-09-08T17:54:29.178045Z",
     "iopub.status.busy": "2020-09-08T17:54:29.177317Z",
     "iopub.status.idle": "2020-09-08T17:54:29.357999Z",
     "shell.execute_reply": "2020-09-08T17:54:29.357384Z"
    },
    "papermill": {
     "duration": 0.227029,
     "end_time": "2020-09-08T17:54:29.358139",
     "exception": false,
     "start_time": "2020-09-08T17:54:29.131110",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAXwAAAEYCAYAAABfgk2GAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjEsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+j8jraAAAgAElEQVR4nO3de1xUdf7H8ddwE028oEA/XS1t8w52oRv+/GkgDloKv8yfZYtm2uattc1K6LaWFWrlbvbz0k9NjS6aPNyFWovLmqVmlFKCyW/LvFbKpcFQAUHm/P5wmR+IgMrA4Jz38/HgAefMOWc+Z46+58x3vud7LIZhGIiIiNvzcHUBIiLSPBT4IiImocAXETEJBb6IiEko8EVETEKBLyJiEgp8E+jduzeRkZGMGDECq9XKlClTOHLkSJM917Fjx+pdZv/+/Xz11VdN8vzN4f7772fjxo215sfGxpKcnHzJ23399dcJDQ0lKiqqxs8TTzxxUdtpbB0AmZmZREZGNmobTz31FK+//nqjtiHO5eXqAqR5JCYmcuWVVwLw6quv8uKLL7J8+XKX1JKRkcGZM2e46aabXPL8LZnVauXFF190dRnipnSGb0K33nprjTP8jz76iDvvvJOoqCgmTJjA4cOHOXPmDDExMaSnpwNw5MgRwsLCyM/PJy4ujoSEBGJjYwkPD+cPf/gDpaWltZ7nrbfeYuTIkURFRTFt2jRsNhubN2/mjTfe4K233mL+/Pm11tm6dStWq5WRI0eyfv16brjhBn788UcyMzO55557eOSRR5g9e3addQPExcWxdOlSxzarT/fu3ZvExESio6MJDw/nvffecyz3/vvvExUVRXh4OI8++ihlZWWOfR87dizDhg1j9uzZVFZW1vnafvfdd9x9990MHTqUp59+msrKSv7whz+watUqxzL//Oc/ufXWWzlz5kzDB6uauLg4/vznPxMbG8utt97KokWL2LBhA6NGjSI8PJzs7Ox66wD4xz/+wahRo7Bardx1113k5uYCnPf1rVJRUUFsbCyrV6+usY2IiAgeeOABbDYbAEVFRTzwwAOEh4fz+9//nhMnTlzU/knTU+CbTHl5OSkpKYSHhwPw888/88wzz7BkyRI+/vhjhg4dyrPPPouXlxfz5s3j5Zdf5vTp08yfP5+ZM2cSGBgIQHp6OosXLyYjIwObzcb7779f43m++eYbVq1aRWJiIh9//DFdunTh1VdfJTw8nMjISCZMmEBcXFyNdSorK4mPj+fJJ59k06ZNHDx4sMYbyd69exk3bhyvvvpqnXVfiIMHD5KcnMzKlSt56aWXsNlsZGdn89prr7F27Vo2b95M27Ztee211wB45ZVXCAsLIyMjg4kTJ5KVlVXntjMzM0lMTOSjjz7iq6++4pNPPuHOO+/kww8/dCyTkZHB8OHD8fK6+A/Yn332meMNc+XKldhsNj744AOsViuJiYn11nHmzBni4uKYN28eqamphIeHs2DBAsc61V/f6l544QV69OjBpEmTOHr0KPHx8bz66qv84x//4JZbbmHu3LkArFixgo4dO7J582aeffZZtm3bdtH7J01LgW8SsbGxREVFMWjQILKzs7nrrrsA2L59O7fccgtXXXUVAGPHjiUzM5OKigqCg4MZOnQos2bN4pdffuHee+91bC88PJyOHTvi4eHBsGHD+Prrr2s835YtW7BarXTq1Mmx3e3bt9db48GDBzl9+jRDhgxx1Gy32x2P+/r6cttttzVYd0PGjBkDQM+ePenRowc5OTl8/PHHREREEBQUBMC9995LWloaADt37mTEiBEAhISE0LNnzzq3bbVaad26Na1bt2bIkCF88803DBkyhMOHD7N//37gbOCPHDnyvOunpqbWasPftGmT4/GwsDDatGnDtddei91u5/bbbwegV69e5Ofn11uHl5cXn3/+Oddddx0AoaGhNT7pVX99q7z33nscPnzY8Wa6efNmgoOD6dWrl+N12rx5M5WVlTVep9/85jfcfPPNdR8EcQm14ZtE9Tb8r776itjYWDZu3EhRURHt2rVzLOfn54dhGBw/fpyAgADGjx/vaFe2WCyO5Tp06OD4u127dhQXF9d4PpvN5vg0ULXML7/8Um+Nv/76K35+fo7p6usDtG/f3vF3fXU3pPp22rdvT3FxMSdOnCA9PZ0vv/wSAMMwHG8ev/76K23btq2xL3Xx9/evUVNBQQGtWrUiMjKSDz/8kLvvvpuCgoI6w7ChNvwrrrgCAIvFgoeHB23atAHAw8Ojxpvj+eqAs/8O/vrXv1JeXk55eXmNY1r9dQEoLCzklVdeITw83PFp5MSJE+zevZuoqCjHcm3btuX48eO1jl99r5O4hgLfhG666Sa6dOnCN998Q6dOnWqcnf/66694eHjQsWNHABYtWsT999/PG2+8wciRIx0BU1RUVGOdc8Oic+fONcL3+PHjdO7cud662rZty6lTpxzThYWFdS5bX93nht/x48fp3r27Y7qoqIiuXbs6Hmvfvj2BgYH853/+J3PmzKn1XO3atePkyZOO6ao26/P59ddfa/xd9brccccdJCQk4Ofnh9VqxcOjaT9cn6+OrKwsVqxYwYYNG/jNb37D9u3beeaZZ+rcho+PD3/961+ZOHEi6enpREZGEhgYSFhYGIsXL661fLt27Wq029tsNrp16+bcHZNGUZOOCR04cIADBw7Qq1cvBg0axM6dOx0f7detW8egQYPw8vJiy5Yt5OXlERcXx+DBg2v8J9+6dSvFxcVUVlaSkZFBaGhojecYOnQo6enpjjeGdevWOZpqvLy8zvuF3tVXX43dbiczMxM425xQ/Qy0uvrqDggI4H//93+Bs1+4ntvc9Pe//x2AH374gUOHDjFw4EDCw8NJS0tzhHlGRgb/8z//A8B1113n+PI6KyvL8eXw+aSlpXH69GlKSkrYunWr43UJCwvj+PHjJCYmOpo9mtL56rDZbHTq1Il/+7d/o7S0lI0bN1JSUlLjzbG6du3a0aVLFxISEnjuueew2Wy1Xvfs7GxeeOEF4OzrlJGRAcDhw4fZtWtXk++nXByd4ZtEbGwsnp6ewNkzt+eee46rr74agHnz5jF9+nTOnDlD165dmTdvHiUlJcybN4/XXnsNi8XCrFmzuOOOOxg1ahRwtqfPzJkzOXz4MCEhIY528SohISH8/ve/57777sNut9O3b1/Hl3u33347jz32GD/99FONNxEfHx/mzp1LfHw8fn5+TJo0CQ8Pj/OG/pVXXnneugH+67/+i5kzZzJ8+HD69euH1Wqtse6bb75JYmIidrudDh06MG7cOLp27crUqVMd3xt06tSJ5557DoDHH3+c2bNnk5yczMCBAwkLC6vzdQ4LC2PChAnk5eUxdOhQBg8eDICnpydRUVFkZGRw44031rl+amrqeYPy448/rnOdC62joqKCd999lyFDhtCtWzeefPJJsrOzmTFjBvfff3+d2woNDeWOO+5g7ty5LF68mHnz5jFjxgwqKiq44oorePLJJwF46KGH+OMf/0h4eDjXXHMNw4cPv6iapelZNB6+XKy4uDi6d+/O9OnTm/R5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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "train_df.boxplot(column=\"Fare\",by = \"Embarked\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "execution_count": null,
   "metadata": {
    "papermill": {
     "duration": 0.04154,
     "end_time": "2020-09-08T17:54:29.441834",
     "exception": false,
     "start_time": "2020-09-08T17:54:29.400294",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "#embarkedda boşluk var mı yok mu yok"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2020-09-08T17:54:29.539584Z",
     "iopub.status.busy": "2020-09-08T17:54:29.538644Z",
     "iopub.status.idle": "2020-09-08T17:54:29.543075Z",
     "shell.execute_reply": "2020-09-08T17:54:29.542449Z"
    },
    "papermill": {
     "duration": 0.059423,
     "end_time": "2020-09-08T17:54:29.543197",
     "exception": false,
     "start_time": "2020-09-08T17:54:29.483774",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Name</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>Embarked</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "Empty DataFrame\n",
       "Columns: [PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked]\n",
       "Index: []"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_df[\"Embarked\"] = train_df[\"Embarked\"].fillna(\"C\")\n",
    "train_df[train_df[\"Embarked\"].isnull()]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2020-09-08T17:54:29.644619Z",
     "iopub.status.busy": "2020-09-08T17:54:29.643468Z",
     "iopub.status.idle": "2020-09-08T17:54:29.647443Z",
     "shell.execute_reply": "2020-09-08T17:54:29.648035Z"
    },
    "papermill": {
     "duration": 0.063047,
     "end_time": "2020-09-08T17:54:29.648188",
     "exception": false,
     "start_time": "2020-09-08T17:54:29.585141",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Name</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>Embarked</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>1026</th>\n",
       "      <td>1044</td>\n",
       "      <td>NaN</td>\n",
       "      <td>3</td>\n",
       "      <td>Storey, Mr. Thomas</td>\n",
       "      <td>male</td>\n",
       "      <td>60.5</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>3701</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "      PassengerId  Survived  Pclass                Name   Sex   Age  SibSp  \\\n",
       "1026         1044       NaN       3  Storey, Mr. Thomas  male  60.5      0   \n",
       "\n",
       "      Parch Ticket  Fare Cabin Embarked  \n",
       "1026      0   3701   NaN   NaN        S  "
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_df[train_df[\"Fare\"].isnull()]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2020-09-08T17:54:29.741418Z",
     "iopub.status.busy": "2020-09-08T17:54:29.740430Z",
     "iopub.status.idle": "2020-09-08T17:54:29.743811Z",
     "shell.execute_reply": "2020-09-08T17:54:29.743063Z"
    },
    "papermill": {
     "duration": 0.053695,
     "end_time": "2020-09-08T17:54:29.743952",
     "exception": false,
     "start_time": "2020-09-08T17:54:29.690257",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "train_df[\"Fare\"] = train_df[\"Fare\"].fillna(np.mean(train_df[train_df[\"Pclass\"] == 3][\"Fare\"]))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2020-09-08T17:54:29.842659Z",
     "iopub.status.busy": "2020-09-08T17:54:29.841809Z",
     "iopub.status.idle": "2020-09-08T17:54:29.845784Z",
     "shell.execute_reply": "2020-09-08T17:54:29.845126Z"
    },
    "papermill": {
     "duration": 0.059012,
     "end_time": "2020-09-08T17:54:29.845913",
     "exception": false,
     "start_time": "2020-09-08T17:54:29.786901",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Name</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>Embarked</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "Empty DataFrame\n",
       "Columns: [PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked]\n",
       "Index: []"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_df[train_df[\"Fare\"].isnull()]"
   ]
  },
  {
   "cell_type": "markdown",
   "execution_count": null,
   "metadata": {
    "papermill": {
     "duration": 0.043679,
     "end_time": "2020-09-08T17:54:29.932223",
     "exception": false,
     "start_time": "2020-09-08T17:54:29.888544",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "#fare ile ilgilide boş bişey kalmadı"
   ]
  },
  {
   "cell_type": "markdown",
   "execution_count": null,
   "metadata": {
    "papermill": {
     "duration": 0.041871,
     "end_time": "2020-09-08T17:54:30.016412",
     "exception": false,
     "start_time": "2020-09-08T17:54:29.974541",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "Visualization(Görselleştirme)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2020-09-08T17:54:30.109528Z",
     "iopub.status.busy": "2020-09-08T17:54:30.108173Z",
     "iopub.status.idle": "2020-09-08T17:54:30.371271Z",
     "shell.execute_reply": "2020-09-08T17:54:30.370624Z"
    },
    "papermill": {
     "duration": 0.312657,
     "end_time": "2020-09-08T17:54:30.371395",
     "exception": false,
     "start_time": "2020-09-08T17:54:30.058738",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "list1 = [\"SibSp\",\"Parch\",\"Age\",\"Fare\",\"Survived\"]\n",
    "sns.heatmap(train_df[list1].corr(), annot = True, fmt = \".2f\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "execution_count": null,
   "metadata": {
    "papermill": {
     "duration": 0.042186,
     "end_time": "2020-09-08T17:54:30.456406",
     "exception": false,
     "start_time": "2020-09-08T17:54:30.414220",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "burda baktığımızda bu gemideki insalar ne kadar para ödediyse hayatta kalma olasılığı o kadar fazla (0.26)"
   ]
  },
  {
   "cell_type": "markdown",
   "execution_count": null,
   "metadata": {
    "papermill": {
     "duration": 0.044679,
     "end_time": "2020-09-08T17:54:30.544227",
     "exception": false,
     "start_time": "2020-09-08T17:54:30.499548",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "SibSp-Survived\n",
    "yeni bir görselleştirme yapıcaz.SbSp ve Survived arasındaki ilişkiyi inceleyeceğiz."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2020-09-08T17:54:30.645428Z",
     "iopub.status.busy": "2020-09-08T17:54:30.637312Z",
     "iopub.status.idle": "2020-09-08T17:54:31.031634Z",
     "shell.execute_reply": "2020-09-08T17:54:31.030990Z"
    },
    "papermill": {
     "duration": 0.444421,
     "end_time": "2020-09-08T17:54:31.031762",
     "exception": false,
     "start_time": "2020-09-08T17:54:30.587341",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<seaborn.axisgrid.FacetGrid at 0x7f01750babd0>"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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4A0UFADASA3u4vvpp5/O+gdbtGjh9jAAAJzWYEE98sgjstlsqqqq0p49e9S6dWudOnVK+/bt07XXXqtly5Z5MicAwM80WFArVqyQJKWlpem1117TlVdeKUnav3+/Fi5c6Jl0AAC/5fLiert373aWkyS1bNlSP/zwwzlNnpGRoc2bN8tmsyktLU2dOnVyLlu2bJmysrIUEBCgmJgYpaene+1k4NqAoDq3AADvc/kvcseOHTVkyBBdf/31stls2rZtm66++mqXE+fn56uoqEiZmZkqLCxUamqqli9fLumXL0H86KOP9N5776lJkyYaNWqUvv76a8XFxV34Gv0GJ1rcoIsObdPJqFivPD8AoD6XBfXUU0/p+++/V2FhoWprazV06FBdc801LifOzc1VYmKiJKl9+/YqKyuTw+FQSEiIfve73+ntt9+W9EtZORwOXXHFFRe4Kr9dddNWqm7aymvPDwCoz+Vh5g6HQ//+97/11VdfqW/fvjp69KjKyspcTmy32xUWFuYch4eHq6SkpM5jXn/9dSUlJalv375q3br1b4gPAGisXG5BTZs2TbfeeqvWrl0rSTpy5IimTp2qN95446x/rra2tt74158xPfjggxo1apTGjRunzp07q3PnzvXm2b59u6uIxjqf7JdamMMKvvy6nI0316uystJ525h+v411veA+HTp0OOP9LguqvLxc99xzjz7++GNJUr9+/fT++++7fMKoqCjZ7XbnuLi4WBEREZJ+ufDsrl27dNNNN+niiy9Wjx49tGnTpjMWVP3gG1w+tyka+qWfyY8W5rDCOa/bZ9bmcLfzec3cLTg42HnrzRzu1ljXC9ZzuYuvpqZGP/74o3PrZ/369aqpqXE5cXx8vHJyciRJBQUFioyMVEhIiCSpurpa06ZNU3l5uSTp22+/VXR09G9eCQBA4+NyC2rmzJmaOXOmtm7dqm7duumaa67RX/7yF5cTx8XFKTY2VsnJybLZbEpPT1d2drZCQ0OVlJSkCRMmaNSoUQoKCtI111yjXr16uWWFAACNg8uCysvL09y5cxUZGXnek6ekpNQZx8TEOH8ePHiwBg8efN5zAgD8g8uCOnr0qB5++GFdfPHF6t27t/r06VPnxF0AAKzg8jOoiRMnasWKFZo/f76CgoI0c+ZM3X333Z7IBgDwY+f0dRsOh0ObNm3S119/rZKSEo7E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\n",
      "text/plain": [
       "<Figure size 432x432 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "g = sns.factorplot(x = \"SibSp\", y=\"Survived\", data = train_df, kind = \"bar\", size = 6)\n",
    "g.set_ylabels(\"Survived Probability\")"
   ]
  },
  {
   "cell_type": "markdown",
   "execution_count": null,
   "metadata": {
    "papermill": {
     "duration": 0.042885,
     "end_time": "2020-09-08T17:54:31.118274",
     "exception": false,
     "start_time": "2020-09-08T17:54:31.075389",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "1.eğer ikiden fazla SibSp değerine sahibse survived probablilty azalıyor.\n",
    "2.sibsp o,1 veya 2 ise passenger has more chance to survive\n",
    "3.yeni bir tane feature extrat edebiliriz\n"
   ]
  },
  {
   "cell_type": "markdown",
   "execution_count": null,
   "metadata": {
    "papermill": {
     "duration": 0.042838,
     "end_time": "2020-09-08T17:54:31.204347",
     "exception": false,
     "start_time": "2020-09-08T17:54:31.161509",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "(bir yolcunun sahip olduğu çocuk veya anne baba asyısı)**Parch--Survived"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2020-09-08T17:54:31.306699Z",
     "iopub.status.busy": "2020-09-08T17:54:31.298354Z",
     "iopub.status.idle": "2020-09-08T17:54:31.756802Z",
     "shell.execute_reply": "2020-09-08T17:54:31.755975Z"
    },
    "papermill": {
     "duration": 0.509155,
     "end_time": "2020-09-08T17:54:31.756940",
     "exception": false,
     "start_time": "2020-09-08T17:54:31.247785",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x432 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "g = sns.factorplot(x = \"Parch\", y = \"Survived\", kind = \"bar\", data = train_df, size=6)\n",
    "g.set_ylabels(\"Survived Probability\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "execution_count": null,
   "metadata": {
    "papermill": {
     "duration": 0.043501,
     "end_time": "2020-09-08T17:54:31.844965",
     "exception": false,
     "start_time": "2020-09-08T17:54:31.801464",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "küçük ailelerin hayatta kalma oranları büyük ailelere göre daha fazladır.gördüğümüz siyah ok sttandart sapma.ortalama değeri kırmızı.yani durumum 1 de olabilir 0 da."
   ]
  },
  {
   "cell_type": "markdown",
   "execution_count": null,
   "metadata": {
    "papermill": {
     "duration": 0.044633,
     "end_time": "2020-09-08T17:54:31.933486",
     "exception": false,
     "start_time": "2020-09-08T17:54:31.888853",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "Sibsp and parch can be used for new feature extraction with th = 3\n",
    "small familes have more chance to survive.\n",
    "there is a std in survival of passenger with parch = 3"
   ]
  },
  {
   "cell_type": "markdown",
   "execution_count": null,
   "metadata": {
    "papermill": {
     "duration": 0.043458,
     "end_time": "2020-09-08T17:54:32.020966",
     "exception": false,
     "start_time": "2020-09-08T17:54:31.977508",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "Pclass -- Survived"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2020-09-08T17:54:32.121749Z",
     "iopub.status.busy": "2020-09-08T17:54:32.114997Z",
     "iopub.status.idle": "2020-09-08T17:54:32.414254Z",
     "shell.execute_reply": "2020-09-08T17:54:32.413617Z"
    },
    "papermill": {
     "duration": 0.349846,
     "end_time": "2020-09-08T17:54:32.414374",
     "exception": false,
     "start_time": "2020-09-08T17:54:32.064528",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAagAAAGoCAYAAAATsnHAAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjEsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+j8jraAAAfaklEQVR4nO3dfVSUdf7/8dcFSG0NqUGQZvalpQKH6ITduWhYgXhK3bRSNm8yy7YttzZ0O4KbbGFsmtZR1043pzLdLVFj2852Q7Vb1vkKB7MiDbYkjdRUZgKiEQ2R+f3xO843AhpAr5kPzPNxTgeuucbreo9z8Nl1MXON5fV6vQIAwDBhwR4AAICOECgAgJEIFADASAQKAGAkAgUAMJLRgdq6dWuwRwAABInRgQIAhC4CBQAwEoECABiJQAEAjESgAABGIlAAACMRKACAkQgUAMBIEXZuvLCwUBUVFbIsS3l5eUpJSZEkHThwQPPmzfPdb/fu3Zo7d67Gjx9v5zgAgF7EtkCVl5erpqZGRUVFqq6uVm5urjZs2CBJiouL09q1ayVJLS0tmj59uq6++mq7RgEA9EK2neIrLS1VRkaGJCkhIUGNjY3yeDzt7vePf/xDWVlZOvXUU+0aBQDQC9l2BOV2u+V0On3L0dHRcrlccjgcbe63YcMGPffcc51up6qqyq4RAQAGSEpK6vB22wL100+S93q9siyrzW0ff/yxzj333HbR+rHOBgcA9G22neKLi4uT2+32LdfW1iomJqbNfd577z2NGDHCrhEAAL2YbYFKS0tTSUmJJKmyslKxsbHtjpS2bdumxMREu0YAAPRitp3iS01NldPpVHZ2tizLUn5+voqLixUVFaXMzExJksvlUnR0tF0jAAB6Mcv7018WGWTr1q0aPnx4sMcAAAQBV5IwUFlZmXJyclRWVhbsUQAgaGy9kgR6ZvXq1dqxY4eampp0xRVXBHscAAgKjqAM1NTU1OYrAIQiAgUAMBKBAgAYiUABAIxEoAAARiJQAAAjESgAgJEIFADASAQKAGAkAgUAMBKBAgAYiUABAIxEoAAARiJQAAAjESgAgJEIFADASAQKAGAkAgUAMBKBAgAYiUABAIxEoAAARiJQAAAjESgAgJEIFADASAQKAGAkAgUAMBKBAgAYiUABAIxEoAAARiJQAAAjESgAgJEigj3AiTT8j2uCPcIJEeX+XuGSvnZ/3+sf09ZHZwR7BAC9FEdQAAAjESgAgJEIFADASAQKAGAkAgUAMBKBAgAYiUABAIxEoAAARiJQAAAjESgAgJFsvdRRYWGhKioqZFmW8vLylJKS4lu3b98+5eTk6MiRIxo2bJgeeughO0cBAPQyth1BlZeXq6amRkVFRVq0aJEKCgrarH/kkUc0a9Ysbdy4UeHh4frmm2/sGgUA0AvZFqjS0lJlZGRIkhISEtTY2CiPxyNJam1t1datW3X11VdLkvLz8zV48GC7RgEA9EK2neJzu91yOp2+5ejoaLlcLjkcDtXV1cnhcGjFihXaunWrLr74YuXk5MiyrHbbqaqqsmtEBADPHwB/kpKSOrzdtkB5vd52y8cC5PV6deDAAd1www265557dMcdd2jTpk0aPXp0u+10NnjHthzHxLBD954/APg/tp3ii4uLk9vt9i3X1tYqJiZGkjRw4EANGjRIQ4cOVXh4uEaMGKEdO3bYNQoAoBeyLVBpaWkqKSmRJFVWVio2NlYOh0OSFBERobPPPltfffWVJOmzzz5TfHy8XaMAAHoh207xpaamyul0Kjs7W5ZlKT8/X8XFxYqKilJmZqby8vKUn5+vH374Qeedd57vBRMAAEg2vw9q3rx5bZYTExN9359zzjlavXq1nbsHAPRiXEkCAGAkAgUAMBKBAgAYiUABAIxEoAAARiJQAAAjESgAgJEIFADASAQKAGAkAgUAMBKBAgAYiUABAIxEoAAARiJQBvKGRbT5CgChiEAZ6PDgi3XEcaYOD7442KMAQNDwv+gGauk/RC39hwR7DAAIKo6gAABGIlAAACMRKACAkQgUAMBIBAoAYCQCBQAwEoECABiJQAGGKCsrU05OjsrKyoI9CmAE3qgLGGL16tXasWOHmpqadMUVVwR7HCDoOIICDNHU1NTmKxDqCBQAwEgECgBgJAIFADASgQIAGIlAAQCMRKAAAEYiUAAAIxEoAICRCBQAwEgECgBgJAIFADASgQIAGIlAAQCMRKAAAEYiUAAAIxEoAICRCBQAwEgECgBgpAg7N15YWKiKigpZlqW8vDylpKT41l1//fWKioryLS9dulRxcXF2jgMA6EVsC1R5eblqampUVFSk6upq5ebmasOGDW3us3btWrt2DwDo5Ww7xVdaWqqMjAxJUkJCghobG+XxeHzrDx48aNeuAQB9gG1HUG63W06n07ccHR0tl8slh8MhSWpoaNDcuXO1d+9eXX755frDH/4gy7LabaeqqsquEREAPH9d19zc7PvK3xtCSVJSUoe32xYor9fbbvnHAbrvvvs0YcIEnXTSSbrrrrv01ltvKSsrq912Ohu8Y1t6Oi5s0r3nL7RFRkb6vvL3Bth4ii8uLk5ut9u3XFtbq5iYGN/yzTffLIfDoX79+mn06NH6/PPP7RoFANAL2RaotLQ0lZSUSJIqKysVGxvrO71XV1en2bNn68iRI5KkLVu26LzzzrNrFABAL2TbKb7U1FQ5nU5lZ2fLsizl5+eruLhYUVFRyszM1OWXX64pU6YoMjJSw4YN6/D0HgAgdNn6Pqh58+a1WU5MTPR9f/vtt+v222+3c/cAgF6MK0kAAIxEoAAARiJQAAAjESgAgJEIFADASAQKAGAkAgUAMBKBAgAYydY36gKB8vVDFwZ7hOPWUne6pAi11NX0icczdOG2YI+AXo4jKACAkQgUAMBIBAoAYCQCBQAwEoECABiJQAEAjESgAABGIlAAACMRKACAkQgUAMBIBAoAYCQCBQAwEoECABiJQAEAjESgAABGIlAAACP5DVRra2sg5gAAoA2/gRozZowWLVqkioqKQMwDAICkLgTq9ddf16hRo/Tyyy9r6tSpWr58ub788stAzAYACGER/u4QGRmp9PR0jRw5Ups3b9aKFSv02muvaciQIcrNzdV5550XiDkBACHGb6DKysr0+uuv66OPPlJaWpr+/Oc/y+l0ateuXZo7d66Ki4sDMScAIMT4DdS6det0/fXXKz8/X+Hh4b7b4+PjNXnyZFuHAwCEri69im/06NFt4nQsTNnZ2fZNBgAIaZ0eQZWUlOjpp5/W559/rhEjRsjr9UqSWlpalJycHLABAQChqdNAZWVlKSsrS88++6xuu+22QM4EAEDngVq3bp2ys7Pldru1ZMmSduvvv/9+WwcDAIS2TgN11llnSZLOP//8gA0DAMAxnQaqtbVVmzZt0umnnx7IeQAAkPQzgXrzzTd/9g+mp6ef8GEAADim00A9+OCDioyM1KFDhwI5DwAAkn4mULm5uVq2bJmuu+46WZblu93r9cqyLP373/8OyIAAgNDUaaCWLVsmSfrPf/4jSaqvr5dlWRowYEBgJgMAhDS/lzoqLi7WihUr5HA4JElNTU3KycnRuHHjbB8OABC6/AbqhRde0CuvvOI7cqqrq9Ott95KoAAAtvJ7Lb5zzjlHp512mm954MCBGjp0qK1DAQDQ6RHU4sWLfS+OuP766zV8+HBZlqVPPvlE8fHxXdp4YWGhKioqZFmW8vLylJKS0u4+y5Yt0yeffKK1a9f28CEAAPqiTgN17AoSP/1AwgsvvFAtLS1+N1xeXq6amhoVFRWpurpaubm52rBhQ5v7VFdXa8uWLerXr19PZgcA9GGdnuKbOHGi77/k5GQNGTJEQ4YMUVxcnNasWeN3w6WlpcrIyJAkJSQkqLGxUR6Pp819HnnkEd13333H+RCAvuHkcG+br0Co8/siiYULF2rnzp3auXOnUlJStH37dt1+++1+N+x2u+V0On3L0dHRcrlcvlcDFhcX67LLLvNd868zVVVVfvcFcwXq+Ts1IHux18T/Oag3d5+isWc3BXuUE4KfXXRVUlJSh7f7DVR1dbVefPFFTZ8+XU8++aT27dunJ554wu8Oj31+1I+Xj/1Oq6GhQcXFxXr++ed14MCBHg3esS3duC8CoXvPX899HZC92Oui6GZdFN0c7DFOmEA99+i7/L6K7+jRo75Tc3V1dRo0aJD++9//+t1wXFyc3G63b7m2tlYxMTGSpLKyMtXV1Wnq1KmaM2eOPvvsMxUWFvb0MQAA+iC/R1DTp0/X66+/rmnTpmn8+PGKiIjQr371K78bTktL08qVK5Wdna3KykrFxsb6Tu+NHTtWY8eOlSTt2bNHubm5ysvLO86HAgDoS/wG6tgbchsaGvTqq68qPDy8S5c7Sk1NldPpVHZ2tizLUn5+voqLixUVFaXMzMzjnxwA0Kd16VJHy5cvV1RUlKTuXepo3rx5bZYTExPb3WfIkCG8BwoA0E6XLnX0z3/+k0sdAQACiksdAQCMZOuljgAA6KkeXeoIAAC7dRqoiRMnSpKam5v1r3/9S5WVlQoPD1dycrKuu+66gA0IAAhNfl8ksWDBAvXv31+pqanyer0qLy9XWVmZHn744UDMBwAIUX4DtX//fj366KO+5euuu04zZsywdSgAAPy+iu/IkSNtrpe3f//+Ln3cBgAAx8PvEVROTo5mzpypsLAwtba2KiwsTAUFBYGYDQAQwvwG6tChQ3rjjTf03XffybKsNu+JAgDALn5P8f3tb39TY2Oj+vfvT5wAAAHj9wjK4/EoPT1dQ4cOVb9+/Xyf67Rx48ZAzAcACFF+A7V06dJAzAEAQBudBqqpqUlr1qxRTU2NLrzwQk2ePFkREX57BgDACdHp76AWLFiglpYWXXvttdq5c6cee+yxQM4FAAhxnR4SuVwuPf7445KkUaNGafr06QEbCgCATo+gwsLarjp2ZXMAAAKh0yOo+vp6bdq0ybfc0NDQZjk9Pd3eyQAAIa3TQCUnJ+vNN9/0LTudzjbLBAoAYKdOA/WXv/wlkHMAANCG3ytJAAAQDAQKAGCkTk/xffPNNz/7BwcPHnzChwEA4JhOA/X73/9elmXpyJEj2rVrl84++2wdPXpUe/bs0bBhw7R+/fpAzgkACDGdBurll1+WJOXl5empp57SmWeeKUnau3evVq5cGZjpAAAhy+/voHbu3OmLkySdddZZ+uqrr+ycCQAA/1czT05O1o033qiLLrpIlmXps88+0/nnnx+I2QAAIcxvoP70pz/pyy+/VHV1tbxer2666SZdcMEFgZgNABDC/J7i83g8evvtt/Xhhx9q7Nixqq+vV2NjYyBmAwCEML+Bmj9/vk477TRt27ZNklRXV6e5c+faPhgAILT5DdTBgwd18803q1+/fpKka6+9VocPH7Z9MABAaPMbqNbWVn399de+j9t4//331draavtgAIDQ5vdFEgsXLtTChQu1fft2jRw5UhdccIEeeuihQMwGAAhhfgNVVlamJUuWKDY2NhDzAAAgqQuBqq+v1+9+9zudfPLJGjNmjLKystq8cRcAADv4/R3UnDlz9PLLL2vZsmWKiIjQwoUL9Zvf/CYQswEAQliXPm7D4/Hoo48+0scffyyXy6WkpCS75wIAhDi/p/huueUWuVwupaena+rUqbr44osDMRcAIMT5DVRubq4SExMDMQsAAD6dBuruu+/WqlWrNHPmTN97oCTJ6/XKsiyVlpYGZEAAQGjqNFCrVq2SJK1Zs4arlwMAAs7vKb6CggI1NDTommuu0dixYzndBwAICL+BWrt2rb777ju99957euKJJ7Rnzx6NHDlSOTk5gZgPABCiuvQy8/79+ystLU2jRo3S4MGDtWnTJrvnAgCEOL9HUKtWrdJ7770ny7KUkZGhuXPnKj4+vksbLywsVEVFhSzLUl5enlJSUnzr1q9fr40bNyosLEyJiYnKz89v82IMAEBo8xuoU045RStWrNCgQYO6teHy8nLV1NSoqKhI1dXVys3N1YYNGyRJhw4d0muvvaa///3v6tevn2bMmKGPP/5YqampPXsUAIA+x+8pvnfffVdnnHFGtzdcWlqqjIwMSVJCQoIaGxvl8XgkSb/4xS/0wgsvqF+/fjp06JA8Hk+P9gEA6Lu6dAQ1ZswYJSYm+j60UJKWL1/+s3/O7XbL6XT6lqOjo+VyueRwOHy3Pf3001qzZo1mzJihs88+u8PtVFVV+X0QMFegnr9TA7IXdAc/u+iqzi6f5zdQs2bN6tEOvV5vu+Wf/o7pjjvu0IwZMzR79mwNHz5cw4cPb7ed7l33b0tPRoWNAnXdxq8Dshd0B9fsxPHyG6jy8vIOb7/ssst+9s/FxcXJ7Xb7lmtraxUTEyNJamho0I4dO3TppZfq5JNP1pVXXqmPPvqow0ABAEKT399BDRw40Pefw+HQF198oe+++87vhtPS0lRSUiJJqqysVGxsrO/0XktLi+bPn6+DBw9KkrZt29blVwYCAEKD3yOoqVOntlmeOXOm7rzzTr8bTk1NldPpVHZ2tizLUn5+voqLixUVFaXMzEzdfffdmjFjhiIiInTBBRfommuu6fmjAAD0OX4DVV1d3WbZ5XJp165dXdr4vHnz2iz/+DJJkyZN0qRJk7q0HQBA6PEbqAcffND3vWVZcjgcysvLs3UoAAC6dC2+Y/bt26fo6GhFRkbaOhQAhIqysjKtX79ekydP1hVXXBHscYzS6YskSktLNX36dEnS0aNHdcstt2jmzJkaN26c3n///YANCAB92erVq1VRUaHVq1cHexTjdHoE9fjjj2vp0qWSpLfeeksej0dvvPGGGhsbdffdd+vKK68M2JAA0Fc1NTW1+Yr/0+kR1EknnaShQ4dKkt5//31NmDBBYWFhGjBggCIi/J4ZBADguHQaqObmZrW2turQoUPatGmTRo0a5VtH6QEAduv0UGjChAmaNGmSmpubNWrUKJ177rlqbm7WAw88oEsuuSSQMwIAQlCngZo6dapGjx6t77//3vf+pcjISF1yySW64YYbAjYgACA0/ewvk84666x2t9100022DQMAwDFd+sh3AAACjUABAIxEoAAARiJQAAAjESgAgJEIFADASAQKAGAkAgUAMBKBAgAYiUABAIxEoAAARiJQAAAj8cmDAHqttJVpwR7huEU2RCpMYdrdsLtPPJ7//f3/nrBtcQQFADASgQIAGIlAAQCMRKAAAEYiUAAAIxEoAICRCBQAwEgECgBgJAIFADASgQIAGIlAAQCMRKAAAEYiUAAAIxEoAICRCBQAwEgECgBgJAIFADASgQIAGIlAAQCMRKAAAEYiUAAAIxEoAAimiJ98hY+tfyWFhYWqqKiQZVnKy8tTSkqKb11ZWZkee+wxhYWFKT4+Xg8//LDCwuglgNDSktSi8OpwHU04GuxRjGNbEcrLy1VTU6OioiItWrRIBQUFbdYvXLhQK1as0Lp163Tw4EF98MEHdo0CAMZqPbNVR0YeUeuZrcEexTi2Baq0tFQZGRmSpISEBDU2Nsrj8fjWFxcX68wzz5QknX766aqvr7drFABAL2TbKT632y2n0+lbjo6OlsvlksPhkCTf19raWm3evFn33ntvh9upqqqya0QEQKCev1MDshd0Bz+7oaknz3tSUlKHt9sWKK/X227Zsqw2t3377be68847tXDhQg0cOLDD7XQ2eMe2dHdM2Kx7z1/PfR2QvaA7AvLcv2P/LtA9J/J5t+0UX1xcnNxut2+5trZWMTExvmWPx6PZs2fr3nvv1ciRI+0aAwDQS9kWqLS0NJWUlEiSKisrFRsb6zutJ0mPPPKIbrnlFqWnp9s1AgCgF7PtFF9qaqqcTqeys7NlWZby8/NVXFysqKgojRw5Uq+88opqamq0ceNGSdK4ceM0ZcoUu8YBAPQytr4Pat68eW2WExMTfd9v377dzl0DAHo53hkLADASgQIAGIlAAQCMRKAAAEYiUAAAIxEoAICRCBQAwEgECgBgJAIFADASgQIAGIlAAQCMRKAAAEYiUAAAIxEoAICRCBQAwEgECgBgJAIFADASgQIAGIlAAQCMRKAAAEYiUAAAIxEoAICRCBQAwEgECgBgJAIFADASgQIAGIlAAQCMRKAAAEYiUAAAIxEoAICRCBQAwEgECgBgJAIFADASgQIAGIlAAQCMRKAAAEYiUAAAIxEoAICRCBQAwEgECgBgJAIFADASgQIAGIlAAQCMRKAAAEayNVCFhYWaMmWKsrOz9emnn7ZZ98MPP+j+++/XpEmT7BwBANBL2Rao8vJy1dTUqKioSIsWLVJBQUGb9UuWLNGwYcPs2j0AoJezLVClpaXKyMiQJCUkJKixsVEej8e3/r777vOtBwDgpyLs2rDb7ZbT6fQtR0dHy+VyyeFwSJIcDocaGhr8bqeqqsquEREAgXr+Tg3IXtAd/OyGpp4870lJSR3eblugvF5vu2XLsrq9nc4G79iWbm8f9ure89dzXwdkL+iOgDz379i/C3TPiXzebTvFFxcXJ7fb7Vuura1VTEyMXbsDAPQxtgUqLS1NJSUlkqTKykrFxsb6Tu8BAOCPbaf4UlNT5XQ6lZ2dLcuylJ+fr+LiYkVFRSkzM1P33HOP9u/fr127dmn69OmaPHmyxo8fb9c4AIBexrZASdK8efPaLCcmJvq+X7FihZ27BgD0clxJAgBgJAIFADASgQIAGIlAAQCMRKAAAEYiUAAAIxEoAICRCBQAwEgECgBgJAIFADASgQIAGIlAAQCMRKAAAEYiUAAAIxEoAICRCBQAwEgECgBgJAIFADASgQIAGIlAAQCMRKAAAEYiUAAAIxEoAICRCBQAwEgECgBgJAIFADASgQIAGIlAAQCMRKAAAEYiUAAAIxEoAICRCBQAwEgECgBgJAIFADASgQIAGIlAAQCMRKAAAEYiUAAAIxEoAICRCBQAwEgECgBgJAIFADASgQIAGIlAAQCMZGugCgsLNWXKFGVnZ+vTTz9ts27z5s268cYbNWXKFK1atcrOMQAAvZBtgSovL1dNTY2Kioq0aNEiFRQUtFm/aNEirVy5Ui+99JI++OADVVdX2zUKAKAXsi1QpaWlysjIkCQlJCSosbFRHo9HkrR79271799fgwYNUlhYmNLT01VaWmrXKACAXijCrg273W45nU7fcnR0tFwulxwOh1wul04//XTfupiYGO3evbvD7WzdurXL+3w62+n/Tgio7jx/x+W61YHZD7rMFYDnfsWvVti+D3RPT3/mhw8f3u422wLl9XrbLVuW1eE6Sb51P9bRwACA0GDbKb64uDi53W7fcm1trWJiYjpcd+DAAZ1xxhl2jQIA6IVsC1RaWppKSkokSZWVlYqNjZXD4ZAkDRkyRB6PR3v27FFLS4veffddpaWl2TUKAKAXsrwdnW87QZYuXaoPP/xQlmUpPz9flZWVioqKUmZmprZs2aKlS5dKksaMGaPbbrvNrjEAAL2QrYFCz3zxxRe66667NHPmTE2bNi3Y4yBAlixZoq1bt6qlpUW//e1vNWbMmGCPBJsdOnRI8+fP17fffqsffvhBd911l6666qpgj2UM214kgZ5pampSQUGBRowYEexREEBlZWXasWOHioqKVF9fr4kTJxKoEPDuu+8qOTlZs2fP1t69ezVr1iwC9SMEyjCRkZF65pln9MwzzwR7FATQpZdeqpSUFElS//79dejQIR09elTh4eFBngx2uvbaa33f79u3T3FxcUGcxjwEyjARERGKiOBpCTXh4eE65ZRTJEkbNmzQlVdeSZxCSHZ2tvbv368nn3wy2KMYhX8JAYO888472rhxo5577rlgj4IAWrdunaqqqvTHP/5Rr776aofvCw1FXM0cMMQHH3ygJ598Us8884yioqKCPQ4CYPv27dq3b58kKSkpSUePHlVdXV2QpzIHgQIM8P3332vJkiV66qmnNGDAgGCPgwD58MMPfUfLbrdbTU1NGjhwYJCnMgcvMzfM9u3btXjxYu3du1cRERGKi4vTypUr+UerjysqKtLKlSsVHx/vu23x4sUaPHhwEKeC3Q4fPqwFCxZo3759Onz4sObMmaOrr7462GMZg0ABAIzEKT4AgJEIFADASAQKAGAkAgUAMBKBAgAYiStJADbYs2ePxo8fr+TkZHm9XjU3N2v27NnKzMxsd9/58+crKyuLi4QCP0GgAJvEx8dr7dq1kqSGhgZNnDhRo0aN0sknnxzkyYDegUABATBgwACdccYZ+vTTT7Vy5UodPXpUgwcP1uLFi3338Xg8mjt3rpqamnT48GE98MADSklJ0dNPP623335bYWFhuuqqq3TnnXd2eBvQ1/A7KCAA9uzZo4aGBq1fv14zZ87Uiy++qNjYWG3fvt13H5fLpZtuuklr165VTk6O7yNXnnvuOb300ktat26dTjvttE5vA/oajqAAm+zatUvTp0+X1+vVSSedpMWLF2vBggVasGCBJOn++++XJL300kuSpJiYGD3xxBN69tln1dzc7Pv4jaysLN16660aN26cJkyY0OltQF9DoACb/Ph3UMeEh4ers6uLvfDCC4qLi9Ojjz6qbdu2acmSJZKkBx98UF9++aXeeOMNTZs2TRs3buzwNj5HDH0Np/iAAEpOTlZZWZkkafny5dq8ebNvXX19vYYOHSrp/38u1JEjR+TxePTXv/5Vv/zlLzVnzhwNGDBAtbW17W7zeDxBeTyAnQgUEED33HOP1q9fr2nTpmnPnj26/PLLfet+/etf6/nnn9esWbOUkpIil8ulkpIS1dfX68Ybb9SMGTN00UUXafDgwe1u42r36Iu4mjkAwEgcQQEAjESgAABGIlAAACMRKACAkQgUAMBIBAoAYCQCBQAw0v8D0GgcbLi4V/kAAAAASUVORK5CYII=\n",
      "text/plain": [
       "<Figure size 432x432 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "g = sns.factorplot(x = \"Pclass\", y = \"Survived\", data = train_df, kind = \"bar\", size = 6)\n",
    "g.set_ylabels(\"Survived Probability\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "execution_count": null,
   "metadata": {
    "papermill": {
     "duration": 0.043399,
     "end_time": "2020-09-08T17:54:32.501690",
     "exception": false,
     "start_time": "2020-09-08T17:54:32.458291",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "x sekseninde pclass var y ekseninde syrvivde prob. birinci sınıftan bilet alan yolcuların ölüm oranı diğerlerine göre daha az"
   ]
  },
  {
   "cell_type": "markdown",
   "execution_count": null,
   "metadata": {
    "papermill": {
     "duration": 0.04339,
     "end_time": "2020-09-08T17:54:32.588650",
     "exception": false,
     "start_time": "2020-09-08T17:54:32.545260",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "Age -- Survived"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2020-09-08T17:54:32.689248Z",
     "iopub.status.busy": "2020-09-08T17:54:32.684336Z",
     "iopub.status.idle": "2020-09-08T17:54:33.152853Z",
     "shell.execute_reply": "2020-09-08T17:54:33.152099Z"
    },
    "papermill": {
     "duration": 0.520283,
     "end_time": "2020-09-08T17:54:33.152975",
     "exception": false,
     "start_time": "2020-09-08T17:54:32.632692",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x216 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "g = sns.FacetGrid(train_df, col = \"Survived\")\n",
    "g.map(sns.distplot, \"Age\", bins = 25)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "execution_count": null,
   "metadata": {
    "papermill": {
     "duration": 0.043889,
     "end_time": "2020-09-08T17:54:33.241283",
     "exception": false,
     "start_time": "2020-09-08T17:54:33.197394",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "hayatta kalnalr kısmında çocukların net olarak kurtulduğunu görüyoruz.15 ve 35 yaş arası çok yolcu var.bu dağılımları kullanarak missing value ları doldurabiliriz"
   ]
  },
  {
   "cell_type": "markdown",
   "execution_count": null,
   "metadata": {
    "papermill": {
     "duration": 0.043611,
     "end_time": "2020-09-08T17:54:33.328993",
     "exception": false,
     "start_time": "2020-09-08T17:54:33.285382",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "age <= 10 has a high survival rate,\n",
    "oldest passengers (80) survived,\n",
    "large number of 20 years old did not survive,\n",
    "most passengers are in 15-35 age range,\n",
    "use age feature in training\n",
    "use age distribution for missing value of age"
   ]
  },
  {
   "cell_type": "markdown",
   "execution_count": null,
   "metadata": {
    "papermill": {
     "duration": 0.044066,
     "end_time": "2020-09-08T17:54:33.417524",
     "exception": false,
     "start_time": "2020-09-08T17:54:33.373458",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "\n",
    "Pclass -- Survived -- Age"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2020-09-08T17:54:33.514044Z",
     "iopub.status.busy": "2020-09-08T17:54:33.513234Z",
     "iopub.status.idle": "2020-09-08T17:54:36.022912Z",
     "shell.execute_reply": "2020-09-08T17:54:36.022325Z"
    },
    "papermill": {
     "duration": 2.561492,
     "end_time": "2020-09-08T17:54:36.023044",
     "exception": false,
     "start_time": "2020-09-08T17:54:33.461552",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 296x432 with 6 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "g = sns.FacetGrid(train_df, col = \"Survived\", row = \"Pclass\", size = 2)\n",
    "g.map(plt.hist, \"Age\", bins = 25)\n",
    "g.add_legend()\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "execution_count": null,
   "metadata": {
    "papermill": {
     "duration": 0.043912,
     "end_time": "2020-09-08T17:54:36.111721",
     "exception": false,
     "start_time": "2020-09-08T17:54:36.067809",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "pclass is important feature for model training."
   ]
  },
  {
   "cell_type": "markdown",
   "execution_count": null,
   "metadata": {
    "papermill": {
     "duration": 0.04423,
     "end_time": "2020-09-08T17:54:36.200353",
     "exception": false,
     "start_time": "2020-09-08T17:54:36.156123",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "burda 6 tane plotumuz mevcut.yolcu sayısını ölenler ve yaşayanları toplayark bulabilir."
   ]
  },
  {
   "cell_type": "markdown",
   "execution_count": null,
   "metadata": {
    "papermill": {
     "duration": 0.044025,
     "end_time": "2020-09-08T17:54:36.288654",
     "exception": false,
     "start_time": "2020-09-08T17:54:36.244629",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "Embarked -- Sex -- Pclass -- Survived"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2020-09-08T17:54:36.398069Z",
     "iopub.status.busy": "2020-09-08T17:54:36.393080Z",
     "iopub.status.idle": "2020-09-08T17:54:37.634624Z",
     "shell.execute_reply": "2020-09-08T17:54:37.633943Z"
    },
    "papermill": {
     "duration": 1.301722,
     "end_time": "2020-09-08T17:54:37.634758",
     "exception": false,
     "start_time": "2020-09-08T17:54:36.333036",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 210.125x432 with 3 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "g = sns.FacetGrid(train_df, row = \"Embarked\", size = 2)\n",
    "g.map(sns.pointplot, \"Pclass\",\"Survived\",\"Sex\")\n",
    "g.add_legend()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "execution_count": null,
   "metadata": {
    "papermill": {
     "duration": 0.044843,
     "end_time": "2020-09-08T17:54:37.724747",
     "exception": false,
     "start_time": "2020-09-08T17:54:37.679904",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "Female passengers have much better survival rate than males.\n",
    "males have better survşval rate in pclass 3 in C.\n",
    "embarked and sex will be used in training."
   ]
  },
  {
   "cell_type": "markdown",
   "execution_count": null,
   "metadata": {
    "papermill": {
     "duration": 0.064373,
     "end_time": "2020-09-08T17:54:37.841226",
     "exception": false,
     "start_time": "2020-09-08T17:54:37.776853",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "#bu plotlara baktığımızda kadınların survive etme oranları erkeklere göre daha yüksek diyebiliriz.cinsiyet ve hayatta kalma birbirleriyle alakalı featurlar.\n",
    "pclass ile embarkes arasında azalış artışlara bakıyoruz yani erkeklerin c limanında hayatta kalma olasılıkları daha yüksek.\n",
    "cinsiyet ve embarkedı doğrudan bir şekilde modelimizi train etmek için kullanacağız."
   ]
  },
  {
   "cell_type": "markdown",
   "execution_count": null,
   "metadata": {
    "papermill": {
     "duration": 0.057273,
     "end_time": "2020-09-08T17:54:37.951710",
     "exception": false,
     "start_time": "2020-09-08T17:54:37.894437",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "Embarked -- Sex -- Fare -- Survived"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2020-09-08T17:54:38.058381Z",
     "iopub.status.busy": "2020-09-08T17:54:38.050799Z",
     "iopub.status.idle": "2020-09-08T17:54:39.651393Z",
     "shell.execute_reply": "2020-09-08T17:54:39.650760Z"
    },
    "papermill": {
     "duration": 1.654362,
     "end_time": "2020-09-08T17:54:39.651540",
     "exception": false,
     "start_time": "2020-09-08T17:54:37.997178",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "image/png": "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S/ufZs2adU23rTKpqanExcVZP9jnz5+nvLycli1b0rRpU2t3ZOfOnZXOP3ToULZu3cqxY8e4++67KzzXs2dPDh06xFdffcXQoUOBS127r7/+GoB//etfpKen4+/vz48//ojFYuHEiRPWrupll7taV/5cneR9+/blq6++ori4mOzsbHJycujYsaP1eXd3d+68806+++47ALZs2cKgQYNuuH1sRXlVN/LqRmojr5y+BXp1Vwtg1qxZ1Zr3zjvvJCYmhqNHjzJjxgy8vLwYMGAAY8aMoUuXLkyaNInFixczYcKEa+Y1mUwkJCQwefJk1q9fT1RUFI888giurq4EBQXRsGFDoqKimDFjBp9++imdO3c29P7atGnDCy+8wKxZs3BxccFkMjFhwoRqHfEMCQnh559/JjQ0lMaNG1NSUkJsbCwNGzZk/PjxLFy4kPbt2/O73/3uutvn2LFjDB48uNL337NnT3766Sdrt+yvf/0rcXFxrFixggYNGvDiiy/SokULOnfuzPjx47njjjvo0qWLoW0wbtw4IiIiMJlMxMfH4+Liwtdff83x48cJDw9n/vz5PP3005SXl9OjR48aHREG5VVV6kteZWdn89RTT5Gbm0tRURFZWVksWLCAkydP1lpeaVC5OuLyuXpGziUUuR7lVc3U6y68iIgtqQUqImKQWqAiIgapgIqIGGTTArp//36CgoKsl4GdOnWKRx99lIiICB599FHr6RADBw6scFOHsrKya5aVmZlpy1DlFqSckpqy2WlMZrOZRYsW0a9fP+u0l19+mXHjxjFixAjWrFnDu+++y6xZs/D19a3XN3YQkfrJZi1QDw8PVqxYUeHM/gULFlivPvDy8uLcuXOYzeZKW5wiIs7OZi1QNze3a24+0LhxY+DSZVhr164lOjoas9nMmTNnmD59Ojk5OYwYMcJ6bfDVfvrpJ1uFK3XIXXfdVWvLUk7JZUbyyu5XIpWVlTF79mz69u1Lv379uHDhAjExMTz00EOUlJQQERFBr169uOeee66ZtzY/OCKgnJKasftR+Hnz5tG+fXvr/QKbNm1KaGgoHh4eNGnShH79+llvBSYi9d/27dv5y1/+wvbt2x0dyk2zawHduHEj7u7uTJ8+3Tpt3759zJkzB4vFQmlpKTt37qRTp072DEtEHCgpKYndu3eTlJTk6FBums268FlZWSxZsoQT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xJtvvsm9996Lm5sb3bt358svv6S8vJzffvuNRYsWWbdXVdshNDT0mu1wdZL379/fOrDgli1bGDRoUIXnnWngQeVV3cmrG6lpXjl9C/RyV+tKs2bNqta8d955JzExMRw9epQZM2bg5eXFgAEDGDNmDF26dGHSpEksXry40qQymUwkJCQwefJk1q9fT1RUFI888giurq4EBQXRsGFDoqKimDFjBp9++imdO3c29P7uuOMO/va3v/Hkk0/SoEEDysrKGD16dLWOWrq4uPDiiy8SHx8PXPrGvv3223n66adp2LAhjRs3JiwsjJ49e1oPJlxt6NChzJ07l7i4uGueCw4OJiEhgddeew241HX9wx/+wPjx47FYLNaDPE8++STz588nOTmZdu3aVVo8bmTatGnMmTOHlJQU2rRpw6hRowD4n//5HxYvXlzrAw8qr66vPuVVYmIi3377Lbm5uTz++OPce++9zJ49u9bySoPKOZE33niDgoICZs+eXenzkZGR9frUG7EN5ZXt1OsufF0TERHBjz/+SHh4OD///LOjw5F6QnllO2qBiogYpBaoiIhBKqAiIgbZtIDu37+foKAg6yVwp06d4tFHHyUiIoJHH33UerXJwIEDK5zUXNkJwJmZmbYMVW5ByimpKZudxmQ2m1m0aFGF87Jefvllxo0bx4gRI1izZg3vvvsus2bNwtfXV0cBRaTOsVkL1MPDgxUrVlS4omTBggXWE2u9vLw4d+4cZrO50haniIizs1kL1M3NDTe3iotv3LgxAGVlZaxdu5bo6GjMZjNnzpxh+vTp5OTkMGLECKKioipd5k8//WSrcKUOueuuu2ptWcopucxIXtn9SqSysjJmz55N37596devHxcuXCAmJoaHHnqIkpISIiIi6NWrF/fcc88189bmB0cElFNSM3Y/Cj9v3jzat2/P1KlTgUu3mwoNDcXDw4MmTZrQr18/663AREScmV0L6MaNG3F3d2f69OnWaZdvmmqxWCgtLWXnzp106tTJnmGJiBhisy58VlYWS5Ys4cSJE7i5ubF582bOnDlDgwYNrDdx6NChA/Hx8bRo0YLQ0FBcXFy4//77rfcVFBFxZnXmUs7MzEzuu+8+R4ch9YhySmpKVyKJiBikAioiYpAKqIiIQSqgIiIGqYCKiBikAioiYpAKqIiIQSqgIiIGqYCKiBikAioiYpAK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\n",
      "text/plain": [
       "<Figure size 339.2x496.8 with 6 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "g = sns.FacetGrid(train_df, row = \"Embarked\", col = \"Survived\", size = 2.3)\n",
    "g.map(sns.barplot, \"Sex\", \"Fare\")\n",
    "g.add_legend()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "execution_count": null,
   "metadata": {
    "papermill": {
     "duration": 0.045522,
     "end_time": "2020-09-08T17:54:39.742764",
     "exception": false,
     "start_time": "2020-09-08T17:54:39.697242",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "1. featurelarımızın arasındaki ilişkiyi inceleyebilriz.İlk önce fare ile survived arasındaki farka bakalım.yani daha çok para harcayanların hayatta kalma olasılığı daha çok.Fare can be used as categorical for training.en yüksek hayatta kalma olasılığı cye en düşük qya orta da s ye ait."
   ]
  },
  {
   "cell_type": "markdown",
   "execution_count": null,
   "metadata": {
    "papermill": {
     "duration": 0.048622,
     "end_time": "2020-09-08T17:54:39.838714",
     "exception": false,
     "start_time": "2020-09-08T17:54:39.790092",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "Passsengers who pay higher fare have better survival. Fare can be used as categorical for training."
   ]
  },
  {
   "cell_type": "markdown",
   "execution_count": null,
   "metadata": {
    "papermill": {
     "duration": 0.045696,
     "end_time": "2020-09-08T17:54:39.930359",
     "exception": false,
     "start_time": "2020-09-08T17:54:39.884663",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "Fill Missing: Age Feature"
   ]
  },
  {
   "cell_type": "markdown",
   "execution_count": null,
   "metadata": {
    "papermill": {
     "duration": 0.045292,
     "end_time": "2020-09-08T17:54:40.021693",
     "exception": false,
     "start_time": "2020-09-08T17:54:39.976401",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "missing valueda yaş kısmında 256 tanelik missing value vardı bunlaro doldurmamız için belli bir analiz gerekiyor o da görselliştirme kısmında oluyor.yani bu kodda age featurenindaki nullara bak ve bunu train dataframinin içinde göster dedim.yani çıktıya göre bizim 256 tane yolcumuzun yaşını bilmiyoruz bunlaru female ve maleların ortalamasına bakarak doldurabilirz ya da pclassa bakarka doldurabilriz.parchye bakıp çocuğu varsa yaşı daha yok yüksek diyebilirz ya da bunları hibrit bir şekilde kullanıp yaş parch gibi hepsine bkaark doldururuz bunlara şimdi bakıcaz"
   ]
  },
  {
   "cell_type": "markdown",
   "execution_count": null,
   "metadata": {
    "papermill": {
     "duration": 0.045295,
     "end_time": "2020-09-08T17:54:40.112681",
     "exception": false,
     "start_time": "2020-09-08T17:54:40.067386",
     "status": "completed"
    },
    "tags": []
   },
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2020-09-08T17:54:40.235741Z",
     "iopub.status.busy": "2020-09-08T17:54:40.220969Z",
     "iopub.status.idle": "2020-09-08T17:54:40.241169Z",
     "shell.execute_reply": "2020-09-08T17:54:40.240567Z"
    },
    "papermill": {
     "duration": 0.082845,
     "end_time": "2020-09-08T17:54:40.241293",
     "exception": false,
     "start_time": "2020-09-08T17:54:40.158448",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Name</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>Embarked</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>6</td>\n",
       "      <td>0.0</td>\n",
       "      <td>3</td>\n",
       "      <td>Moran, Mr. James</td>\n",
       "      <td>male</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>330877</td>\n",
       "      <td>8.4583</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Q</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>17</th>\n",
       "      <td>18</td>\n",
       "      <td>1.0</td>\n",
       "      <td>2</td>\n",
       "      <td>Williams, Mr. Charles Eugene</td>\n",
       "      <td>male</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>244373</td>\n",
       "      <td>13.0000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>19</th>\n",
       "      <td>20</td>\n",
       "      <td>1.0</td>\n",
       "      <td>3</td>\n",
       "      <td>Masselmani, Mrs. Fatima</td>\n",
       "      <td>female</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>2649</td>\n",
       "      <td>7.2250</td>\n",
       "      <td>NaN</td>\n",
       "      <td>C</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>26</th>\n",
       "      <td>27</td>\n",
       "      <td>0.0</td>\n",
       "      <td>3</td>\n",
       "      <td>Emir, Mr. Farred Chehab</td>\n",
       "      <td>male</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>2631</td>\n",
       "      <td>7.2250</td>\n",
       "      <td>NaN</td>\n",
       "      <td>C</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>27</th>\n",
       "      <td>29</td>\n",
       "      <td>1.0</td>\n",
       "      <td>3</td>\n",
       "      <td>O'Dwyer, Miss. Ellen \"Nellie\"</td>\n",
       "      <td>female</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>330959</td>\n",
       "      <td>7.8792</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Q</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1282</th>\n",
       "      <td>1300</td>\n",
       "      <td>NaN</td>\n",
       "      <td>3</td>\n",
       "      <td>Riordan, Miss. Johanna Hannah\"\"</td>\n",
       "      <td>female</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>334915</td>\n",
       "      <td>7.7208</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Q</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1284</th>\n",
       "      <td>1302</td>\n",
       "      <td>NaN</td>\n",
       "      <td>3</td>\n",
       "      <td>Naughton, Miss. Hannah</td>\n",
       "      <td>female</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>365237</td>\n",
       "      <td>7.7500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Q</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1287</th>\n",
       "      <td>1305</td>\n",
       "      <td>NaN</td>\n",
       "      <td>3</td>\n",
       "      <td>Spector, Mr. Woolf</td>\n",
       "      <td>male</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>A.5. 3236</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1290</th>\n",
       "      <td>1308</td>\n",
       "      <td>NaN</td>\n",
       "      <td>3</td>\n",
       "      <td>Ware, Mr. Frederick</td>\n",
       "      <td>male</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>359309</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1291</th>\n",
       "      <td>1309</td>\n",
       "      <td>NaN</td>\n",
       "      <td>3</td>\n",
       "      <td>Peter, Master. Michael J</td>\n",
       "      <td>male</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2668</td>\n",
       "      <td>22.3583</td>\n",
       "      <td>NaN</td>\n",
       "      <td>C</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>256 rows × 12 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "      PassengerId  Survived  Pclass                             Name     Sex  \\\n",
       "5               6       0.0       3                 Moran, Mr. James    male   \n",
       "17             18       1.0       2     Williams, Mr. Charles Eugene    male   \n",
       "19             20       1.0       3          Masselmani, Mrs. Fatima  female   \n",
       "26             27       0.0       3          Emir, Mr. Farred Chehab    male   \n",
       "27             29       1.0       3    O'Dwyer, Miss. Ellen \"Nellie\"  female   \n",
       "...           ...       ...     ...                              ...     ...   \n",
       "1282         1300       NaN       3  Riordan, Miss. Johanna Hannah\"\"  female   \n",
       "1284         1302       NaN       3           Naughton, Miss. Hannah  female   \n",
       "1287         1305       NaN       3               Spector, Mr. Woolf    male   \n",
       "1290         1308       NaN       3              Ware, Mr. Frederick    male   \n",
       "1291         1309       NaN       3         Peter, Master. Michael J    male   \n",
       "\n",
       "      Age  SibSp  Parch     Ticket     Fare Cabin Embarked  \n",
       "5     NaN      0      0     330877   8.4583   NaN        Q  \n",
       "17    NaN      0      0     244373  13.0000   NaN        S  \n",
       "19    NaN      0      0       2649   7.2250   NaN        C  \n",
       "26    NaN      0      0       2631   7.2250   NaN        C  \n",
       "27    NaN      0      0     330959   7.8792   NaN        Q  \n",
       "...   ...    ...    ...        ...      ...   ...      ...  \n",
       "1282  NaN      0      0     334915   7.7208   NaN        Q  \n",
       "1284  NaN      0      0     365237   7.7500   NaN        Q  \n",
       "1287  NaN      0      0  A.5. 3236   8.0500   NaN        S  \n",
       "1290  NaN      0      0     359309   8.0500   NaN        S  \n",
       "1291  NaN      1      1       2668  22.3583   NaN        C  \n",
       "\n",
       "[256 rows x 12 columns]"
      ]
     },
     "execution_count": 37,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_df[train_df[\"Age\"].isnull()]"
   ]
  },
  {
   "cell_type": "markdown",
   "execution_count": null,
   "metadata": {
    "papermill": {
     "duration": 0.045439,
     "end_time": "2020-09-08T17:54:40.332575",
     "exception": false,
     "start_time": "2020-09-08T17:54:40.287136",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "x ekseni cinsiyet y ekseni yaş.burdaki yaş sınırlarına bakarsak medyan değerleri aynı gibi yani burdan kadın ve erkek olmasından bir çıkarım yapamayız(aşağının açıklaması bu) 38.kod"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2020-09-08T17:54:40.442518Z",
     "iopub.status.busy": "2020-09-08T17:54:40.431812Z",
     "iopub.status.idle": "2020-09-08T17:54:40.637130Z",
     "shell.execute_reply": "2020-09-08T17:54:40.636317Z"
    },
    "papermill": {
     "duration": 0.258565,
     "end_time": "2020-09-08T17:54:40.637260",
     "exception": false,
     "start_time": "2020-09-08T17:54:40.378695",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAWAAAAFgCAYAAACFYaNMAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjEsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+j8jraAAAb7ElEQVR4nO3dfXST9d3H8U9oGmjLRFpStFXA0blWwE2cf1Atx7MAFZzHypmz5oAcz8HJYKU6BwUsioIoDz7AZLI5ZVCs64wMUSqp7OjmPLFqYWrvFUdVQARLaXlOSmnI/Qc3uccoQqFXfkn6fv3VXiXX75s0vr169UpqC4VCIQEAIq6b6QEAoKsiwABgCAEGAEMIMAAYQoABwJCoC3BNTY3pEQAgIqIuwADQVRBgADCEAAOAIQQYAAwhwABgCAEGAEMIMAAYQoABwBACDACGEGAAMMSyAB85ckS//OUvNX78eBUWFurdd9/V7t27NX78eLndbhUXF6u1tdWq5buMpqYmTZ06VU1NTaZHAdBBlgX4L3/5i6644gqVlZVpyZIleuyxx7R06VK53W6Vl5crMzNTHo/HquW7jJUrV+rTTz/VqlWrTI8CoIMsC3Dv3r21f/9+SdLBgwfVu3dvVVdXy+VySZJcLpd8Pp9Vy3cJTU1N2rBhg0KhkDZs2MBRMBBjLAvwzTffrF27dmnkyJEaN26cSkpKFAgE5HA4JElOp1ONjY1WLd8lrFy5UsePH5ckBYNBjoKBGGO3asevvfaaMjIy9MILL2jLli168MEHZbPZwl//tr8FWldXZ9VYcaWqqkptbW2SpLa2Nnm9Xt10002GpwLw33JyctrdblmAN23apBtuuEGSlJ2drYaGBiUlJamlpUU9evRQQ0OD0tPTOzQsTjVq1ChVVlaqra1Ndrtd+fn5PHZADLHsFET//v318ccfS5K+/vprpaSkKDc3V16vV9KJo7e8vDyrlu8SJkyYoG7dTnwLExISdNdddxmeCEBHWBbgO+64Q19//bXGjRunBx54QHPmzFFRUZHWrl0rt9ut/fv3q6CgwKrlu4S0tDTddNNNstlsuummm5SWlmZ6JAAdYAt928lYA2pqanTttdeaHiNmNDU16ZFHHtHDDz9MgIEYQ4ABwBBeigwAhhBgADCEAAOAIQQYAAwhwADOCe+81/kIMIBzwjvvdT4CDOCseOc9axBgAGfFO+9ZgwADOKuNGzee8s57b731luGJ4gMBBnBWI0aMkN1+4s0T7Xa7Ro4caXii+ECAAZwV77xnDQIc47g0CJHAO+9ZgwDHOC4NQqRMmDBBQ4YM4ei3ExHgGMalQYiktLQ0LV26lKPfTkSAYxiXBgGxjQDHMC4NAmIbAY5hXBoExDYCHMO4NAiIbQQ4hnFpEBDb7KYHwIWZMGGCtm3bxtEvEIP4o5wAYAinIGJcfX29br75ZtXX15seBUAHEeAYN2/ePB05ckTz5s0zPQqADiLAMay+vl7btm2TJG3bto2jYCDGEOAY9t9HvRwFA7GFAMewk0e/Z/ocQHSz7DK0V155RevWrQt/Xltbq8rKSk2fPl3BYFBOp1OLFi2Sw+GwaoS417NnTx0+fPiUzwHEjohchvbBBx/ozTffVEtLi4YPH67Ro0dr4cKFuuyyy+R2u0/5t1yGdu5GjRql1tbW8OcOh0NVVVUGJwLQERE5BbFs2TJNnjxZ1dXVcrlckiSXyyWfzxeJ5ePWf//0wE8TQGyx/JVwn3zyiS699FI5nU4FAoFwJJxOpxobG9u9TV1dndVjxYX/PP1w8nMeOyD65OTktLvd8gB7PB7ddtttkiSbzRbe/m1nPs40LE41YMCAU37xNmDAAB47IIZYfgqiurpa11xzjSQpKSlJLS0tkqSGhgalp6dbvXxcKy0t/dbPAUQ3SwPc0NCglJSU8GmH3Nxceb1eSVJVVZXy8vKsXD7uZWVlacCAAZJOHP1mZWWZHQhAh1ga4MbGRqWmpoY/Lyoq0tq1a+V2u7V//34VFBRYuXyXUFpaqpSUFI5+gRjEu6EBgCG8Eg4ADCHAMa6pqUlTp07lT9IDMYgAx7iVK1fq008/5U/SAzGIAMewpqYmbdiwQaFQSBs2bOAoGIgxBDiGrVy5UsePH5ckBYNBjoKBGEOAY9jGjRvV1tYmSWpra9Nbb71leCIAHUGAY9iIESNkt594NbndbtfIkSMNTwSgI/iz9J3A6/WqsrIy4useO3YsfAQcDAa1detWFRcXR3SGMWPGKD8/P6JrAvGCI+AYlpiYGD4CTk1NVWJiouGJAHQEr4SLcZMnT9b27du1atUqpaWlmR4HQAdwBBzjEhMTlZWVRXyBGESAAcAQAgwAhhBgADCEy9CAGGLqkkdJ2rdvnySpd+/eRtaPx0seCTCAc3LyvUZMBTgeEWAghuTn5xs7Cjz5Ip8lS5YYWT8ecQ4YAAwhwABgCAEGAEMIMAAYQoABwBACDACGEGAAMIQAA4AhBBgADLH0lXDr1q3TH/7wB9ntdhUXF+vKK6/U9OnTFQwG5XQ6tWjRIjkcDitHAICoZdkR8L59+7Rs2TKVl5dr+fLl2rhxo5YuXSq3263y8nJlZmbK4/FYtTwARD3LAuzz+TRs2DD17NlT6enpmjt3rqqrq+VyuSRJLpdLPp/PquUBIOpZdgpi586dCoVCuu+++7Rnzx4VFRUpEAiETzk4nU41Nja2e9u6ujqrxoo7fr9fEo8ZrMdz7fzl5OS0u93Sc8ANDQ169tlntWvXLt11112y2Wzhr33b3wI907A4XXJysiQeM1iP51rns+wURFpamq655hrZ7Xb169dPKSkpSkpKUktLi6QTcU5PT7dqeQCIepYF+IYbbtD777+v48ePq7m5WX6/X7m5ufJ6vZKkqqoq5eXlWbU8AEQ9y05B9O3bV/n5+ZowYYICgYBKS0s1ZMgQlZSUqKKiQhkZGSooKLBqeQCIepaeAy4sLFRhYeEp21asWGHlkgAQM3glHAAYQoABwBACDACGEGAAMIQAA4AhBBgADCHAAGAIAQYAQwgwABhCgAHAEAIMAIYQYAAwhAADgCEEGAAMIcAAYAgBBgBDCDAAGEKAAcAQAgwAhhBgADCEAAOAIQQYAAwhwABgCAEGAEMIMAAYQoABwBACDACG2K3acW1trSZPnqz+/ftLkq688kpNnDhR06dPVzAYlNPp1KJFi+RwOKwaAQCimmUB9vv9ys/P14MPPhjeNnPmTLndbo0ePVoLFy6Ux+OR2+22agQAiGqWnYI4cuTIaduqq6vlcrkkSS6XSz6fz6rlASDqWXoEXFNTo4kTJyoQCKioqEiBQCB8ysHpdKqxsbHd29bV1Vk1Vtzx+/2SeMxgPZ5r5y8nJ6fd7ZYFODs7W1OmTJHL5dKXX36pu+++W21tbeGvh0KhM972TMPidMnJyZJ4zGA9nmudz7IADxw4UAMHDpQkXXHFFerTp492796tlpYW9ejRQw0NDUpPT7dqeQCIepadA/Z4PFq1apUkqbGxUU1NTRo7dqy8Xq8kqaqqSnl5eVYtDwBRz7Ij4JEjR+rXv/61vF6vWltbNWfOHOXk5KikpEQVFRXKyMhQQUGBVcsDQNSzLMC9evXS888/f9r2FStWWLUkAMQUXgkHAIYQYAAwhAADgCEEGAAMIcAAYAgBBgBDCDAAGEKAAcAQAgwAhhBgADCEAAOAIQQYAAwhwABgCAEGAEMIMAAYQoABwBACDACGEGAAMIQAA4AhBBgADCHAAGAIAQYAQwgwABhyTgFubW3Vzp07rZ4FALqUswZ4/fr1Gjt2rCZNmiRJmjdvntauXWv5YAAQ784a4Jdeeklr1qxR7969JUnTpk1TeXm55YMBQLw7a4ATEhLkcDhks9kkSQ6H45x33tLSIpfLpTVr1mj37t0aP3683G63iouL1draev5TA0AcOGuAhw4dqmnTpqmhoUG///3vdeedd2rYsGHntPPnnntOF198sSRp6dKlcrvdKi8vV2Zmpjwez4VNDgAx7qwBvv/++3XHHXfopz/9qbp3766SkhLdf//9Z93x559/rvr6et14442SpOrqarlcLkmSy+WSz+e7sMkBIMbZz/YPnn322fDHR48e1Xvvvafq6mr169dPo0aNkt3e/i4WLFig2bNnh39hFwgEwqcvnE6nGhsbz7hmXV1dh+5EV+b3+yXxmMF6PNfOX05OTrvbzxpgv9+vTZs2afjw4erWrZvee+89DRw4ULt27VJVVZWeeeaZ026zdu1a/fCHP9Tll18e3nbyHLIkhUKh8xoWp0tOTpbEYwbr8VzrfGcN8GeffaaXX345HNB77rlHU6ZM0fLlyzVu3Lh2b/POO+/oq6++0jvvvKNvvvlGDodDSUlJamlpUY8ePdTQ0KD09PTOvScAEGPOGuA9e/bos88+U3Z2tiRpx44d2rlzp3bt2qUjR460e5v/PCr+zW9+o8zMTG3evFler1e33nqrqqqqlJeX10l34f/Xqa+v79R9xoKT97m4uNjwJJGXlZWloqIi02MA5+2sAZ45c6ZmzZql3bt3SzpxLvcXv/iFvvzySz3wwAPnvFBRUZFKSkpUUVGhjIwMFRQUnP/U7aivr9c/a+sUTE7t1P1GO1vwxLew5osGw5NEVoK/2fQIwAU7a4Bzc3P13HPP6c0339T69et14MABHT9+XNdff/05LfCfRygrVqw4/0nPQTA5VYHsMZaugeiQtKXS9AjABTtjgPfv3y+v16s33nhD27dv16hRo3To0CFVVVVFcj4g6nC6i9NdneWMAb7hhhvUr18/lZSUKC8vT926dev00wZALKqvr9fW/9msfj2DpkeJqItCJ34Rf3T7R4YniawdhxMs2/cZA/z4449r/fr1mjVrln784x9rzBh+tAdO6tczqFlDD5oeAxEwf9NFlu37jK+Eu+WWW7R8+XJVVlZq0KBBWrZsmb744gstWLCgS/74BQCd7awvRe7Vq5cKCwu1evVqVVVVKS0tTdOnT4/EbAAQ1zr0FzEuueQSTZw4UWvWrLFqHgDoMviTRABgCAEGAEMIMAAYQoABwBACDACGEGAAMIQAA4AhBBgADCHAAGAIAQYAQwgwABhCgAHAEAIMAIYQYAAwhAADgCEEGAAMIcAAYAgBBgBDCDAAGEKAAcAQu1U7DgQCmjFjhpqamnT06FFNnjxZ2dnZmj59uoLBoJxOpxYtWiSHw2HVCAAQ1Sw7An777bc1ePBgrV69Ws8884yeeOIJLV26VG63W+Xl5crMzJTH47FqeQCIepYFeMyYMbrnnnskSbt371bfvn1VXV0tl8slSXK5XPL5fFYtDwBRz7JTECcVFhbqm2++0fLly3X33XeHTzk4nU41NjZavTwARC3LA/ynP/1JdXV1mjZtmmw2W3h7KBQ6423q6uo6vI7f7z+v+RC7/H7/eT1XOmPdhIivCpMu9LmWk5PT7nbLAlxbW6u0tDRdeumlysnJUTAYVFJSklpaWtSjRw81NDQoPT29Q8N+m+TkZEmHLnBqxJLk5OTzeq50xrpHI74qTLLquWbZOeCPPvpIL774oiRp79698vv9ys3NldfrlSRVVVUpLy/PquUBIOpZdgRcWFioBx98UG63Wy0tLXrooYc0ePBglZSUqKKiQhkZGSooKLBqeQCIepYFuEePHnryySdP275ixQpL1mtublaCv0lJWyot2T+iS4K/Sc3NiabHAC4Ir4QDAEMsvwoiUlJTU/Xl/mMKZI8xPQoiIGlLpVJTU02PAVwQjoABwBACDACGxM0pCCBSmpubtfdQguZvusj0KIiA7YcS1Ke52ZJ9cwQMAIZwBAx0UGpqqlIOfaFZQw+aHgURMH/TRepu0S98OQIGAEMIMAAYQoABwBACDACGEGAAMIQAA4AhBBgADCHAAGAIAQYAQwgwABhCgAHAEAIMAIYQYAAwhAADgCEEGAAMIcAAYAgBBgBDCDAAGBJXf5Iowd+spC2VpseIKNuxgCQplJhkeJLISvA3S+pregzggsRNgLOyskyPYER9fb0kKeu7XS1Gfbvs9xzxw9IAL1y4UDU1NWpra9O9996rIUOGaPr06QoGg3I6nVq0aJEcDkenrFVUVNQp+4k1xcXFkqQlS5YYngRAR1kW4Pfff19bt25VRUWF9u3bp9tuu03Dhg2T2+3W6NGjtXDhQnk8HrndbqtGAICoZtkv4a677rrwUVmvXr0UCARUXV0tl8slSXK5XPL5fFYtDwBRz7Ij4ISEBCUnJ0uSXnnlFQ0fPlz/+Mc/wqccnE6nGhsb271tXV2dVWPFHb/fL4nHLJL8fr8STA+BiPL7/Rf031hOTk672y3/JdzGjRvl8Xj04osvKj8/P7w9FAqd8TZnGhanO/k/OR6zyElOTtZR00MgopKTky35b8zS64DfffddLV++XM8//7y+853vKCkpSS0tLZKkhoYGpaenW7k8AEQ1y46ADx06pIULF+qPf/yjLr74YklSbm6uvF6vbr31VlVVVSkvL8+q5QFL7TicoPmbLjI9RkQdaLVJkno5zvzTazzacThB37No35YFuLKyUvv27dN9990X3vbEE0+otLRUFRUVysjIUEFBgVXLA5bpqtcfH/y/a87T+3et+/89Wfc9t4W+7WSsATU1Nbr22mtNjxEzuA4YkcJzrfPxXhAAYAgBBgBDCDAAGEKAAcAQAgwAhhBgADCEAAOAIQQYAAwhwABgCAEGAEMIMAAYQoABwBACDACGEGAAMIQAA4AhBBgADCHAAGAIAQYAQwgwABhCgAHAEAIMAIYQYAAwhAADgCEEGAAMIcAAYAgBBgBDLA3wv//9b40YMUKrV6+WJO3evVvjx4+X2+1WcXGxWltbrVweAKKaZQH2+/2aO3euhg0bFt62dOlSud1ulZeXKzMzUx6Px6rlASDqWRZgh8Oh559/Xunp6eFt1dXVcrlckiSXyyWfz2fV8gAQ9eyW7dhul91+6u4DgYAcDockyel0qrGxsd3b1tXVWTVW3PH7/ZJ4zGA9nmvnLycnp93tlgW4PTabLfxxKBQ6478707A4XXJysiQeM1iP51rni+hVEElJSWppaZEkNTQ0nHJ6AgC6mogGODc3V16vV5JUVVWlvLy8SC4PAFHFslMQtbW1WrBggb7++mvZ7XZ5vV4tXrxYM2bMUEVFhTIyMlRQUGDV8gAQ9SwL8ODBg1VWVnba9hUrVli1JADEFF4JBwCGEGAAMIQAA4AhBBgADCHAAGAIAQYAQwgwABhCgAHAEAIMAIYQYAAwhAADgCEEGAAMIcAAYAgBBgBDCDAAGEKAAcAQAgwAhhBgADCEAAOAIQQYAAwhwABgCAEGAEMIMAAYQoABwBACDACGEGAAMIQAA4Ah9kgvOH/+fH388cey2WyaNWuWrr766kiPAABRIaIB/uCDD7R9+3ZVVFSovr5eM2fO1CuvvBLJESzh9XpVWVlpZO36+npJUnFxsZH1x4wZo/z8fCNrd0U81+LruRbRAPt8Po0YMUKSlJWVpYMHD+rw4cPq2bNnJMeIK2lpaaZHQBfBc63zRTTAe/fu1aBBg8Kfp6WlqbGx8bQA19XVRXKsC9avXz9NmjTJ9BjGxNr3K5bxXIvN51pOTk672yMa4FAodNrnNpvttH93pmEBIJ5E9CqIvn37au/eveHP9+zZoz59+kRyBACIGhEN8PXXXy+v1ytJ+te//qX09HTO/wLosiJ6CmLo0KEaNGiQCgsLZbPZ9PDDD0dyeQCIKrbQf5+YNaympkbXXnut6TEAwHK8Eg4ADCHAAGAIAQYAQwgwABhCgAHAEAIMAIYQYAAwJOLvB3wuampqTI8AAJ2qvdc3RN0LMQCgq+AUBAAYQoABwBACDACGEOA4M2PGDL399tumx0AUO3bsmG6//XaVlJR02j537typsWPHdtr+ugoCDHQxjY2Nam1t1YIFC0yP0uVF5WVoOGHNmjX68MMPtW/fPm3dulX333+/3njjDX3++edavHixKisr9cknn+jo0aO68847dfvtt4dvGwwGNXv2bH311Vdqa2vT1KlTNWzYMIP3BtHi8ccf144dOzRz5kwdOXJEBw4cUDAYVGlpqbKzszVixAj97Gc/04YNG9S/f38NGjQo/PGTTz6pLVu26JFHHpHdble3bt20ZMmSU/b/0Ucf6amnnpLdbtell16quXPnyuFwGLq30Y0j4Ci3bds2Pffcc7r33nv1u9/9TsuWLdPPf/5zvfrqq8rMzNTLL7+s8vLy0/4jeP311+V0OlVWVqZly5Zp/vz5hu4Bok1JSYmuuOIKXXbZZcrLy9PKlSs1Z86c8BHx8ePHddVVV+nVV1/Vpk2blJmZKY/Ho5qaGh08eFBNTU2aPXu2ysrKNHToUL3++uun7H/evHn67W9/q1WrViktLU0bNmwwcTdjAkfAUW7w4MGy2WxyOp36/ve/r4SEBPXp00fHjh3TgQMHVFhYqMTERO3bt++U223evFk1NTXatGmTJOno0aNqbW3lSARhmzdvVnNzs9atWydJCgQC4a9dffXVstlsSktL01VXXSVJSk1N1aFDh5SWlqbFixerpaVFe/bs0S233BK+3d69e7V9+3YVFRVJkvx+v3r37h3BexVbCHCUs9vt7X68c+dO7dixQ2VlZUpMTNQ111xzyu0SExM1adIk/eQnP4nYrIgtiYmJmj179mnPHUlKSEho9+NQKKTHHntM99xzj4YPH64XXnhBfr//lH2mp6errKzM2uHjBKcgYlRtba0uueQSJSYm6q9//auCwaBaW1vDX//BD36gjRs3SpKampr01FNPmRoVUeo/nyP19fVasWLFOd1u//796tevn1pbW/W3v/1Nx44dC3+tV69e4f1JUllZmbZs2dLJk8cPAhyjcnNztX37do0bN05fffWVbrzxRs2ZMyf89dGjRyslJUWFhYWaNGkSf2cPpxk3bpx27Nght9ut0tJS/ehHPzrn202ZMkVTp07V+PHjtXbtWh0+fDj89ccee0wzZ86U2+1WTU2Nvvvd71p1F2Ie7wUBAIZwBAwAhhBgADCEAAOAIQQYAAwhwABgCC/EQFx76aWX9Nprr6l79+4KBAL61a9+pdzcXNNjAZIIMOLYzp079ec//1kej0eJiYnatm2bSktLCTCiBqcgELcOHz6so0ePhl+pNWDAAK1evVr19fW66667NGHCBE2ePFkHDx7Uhx9+qEmTJkk68W5eEydONDk6uggCjLiVnZ2tq6++Wi6XSzNmzFBlZaXa2to0d+5cPfroo1q5cqWuv/56vfTSS7ruuut08cUX67333tPTTz+thx56yPT46AJ4JRzi3ueff653331X69atU0pKimprazV48GBJUmtrq4YMGaLS0lI1Nzfr9ttv19ixYzVlyhTDU6Mr4Bww4lYoFFJra6sGDhyogQMHavz48Ro9erT8fr9WrVolm812yr8/fPiwHA6HGhoaDE2MroZTEIhbHo9Hs2fP1skf8g4dOqTjx48rNzdXf//73yVJ69evl8/nk3TijcSffvpp7dmzR//85z+NzY2ug1MQiFvBYFCLFy/Whx9+qOTkZB07dkz33nuvLr/8cs2ePVvdunVT9+7d9eSTT8rn88nn8+nRRx/VF198oWnTpqmiouKU92AGOhsBBgBDOAUBAIYQYAAwhAADgCEEGAAMIcAAYAgBBgBDCDAAGPK/nQR2qznB2V8AAAAASUVORK5CYII=\n",
      "text/plain": [
       "<Figure size 360x360 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.factorplot(x = \"Sex\", y = \"Age\", data = train_df, kind = \"box\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "execution_count": null,
   "metadata": {
    "papermill": {
     "duration": 0.046354,
     "end_time": "2020-09-08T17:54:40.729934",
     "exception": false,
     "start_time": "2020-09-08T17:54:40.683580",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "Sex is not informative for age prediction, age distribution seems to be same.\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2020-09-08T17:54:40.831757Z",
     "iopub.status.busy": "2020-09-08T17:54:40.830585Z",
     "iopub.status.idle": "2020-09-08T17:54:41.196452Z",
     "shell.execute_reply": "2020-09-08T17:54:41.195676Z"
    },
    "papermill": {
     "duration": 0.420272,
     "end_time": "2020-09-08T17:54:41.196577",
     "exception": false,
     "start_time": "2020-09-08T17:54:40.776305",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "image/png": "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AAwRFgAAQ4QFAMAQYQEAMERYAAAMERYAAEOsOgsAFlFXV6fbb79dycnJcrvdcrlcuvfeezVlypTztl28eLGmTp2qm2++2S+1ERYAYCFjxoxRYWGhJOn48eO68847lZqaqsjIyIDWRVgAgEUNGzZMcXFx2rVrl9atW6fOzk4lJCRo1apVnm1aWlr0yCOPyOl0qq2tTbm5uZowYYJ+9atf6a233lJoaKhuvvlm3XfffT3e5y2OWQCARdXV1en48eP67W9/qx/+8IcqKipSfHy8du/e7dmmoaFBM2bMUGFhoR5++GFt3LhRkvTyyy/rtdde029+8xt96Utf6vU+bzGyAAAL2bdvn2bPni23262BAwdq1apVWrp0qZYuXSpJWrRokSTptddekyTFxsbql7/8pV566SW5XC5FRUVJkqZOnaq7775bt912m77zne/0ep+3giosmpqaFOpstPSlJEOdjWpqigh0GQAs6ovHLM4ZMGCA3G53j9sXFBRoxIgRWrNmjf72t79p9erVkqQVK1Zo79692rZtm2bNmqWSkpIe7wsL8y4G2A0FABaXnJysHTt2SJKeffZZ/c///I/nZ83NzRo1apQkafv27Wpvb1dLS4uee+45jR07Vg888ICGDRumo0ePnndfS0uL1zV4FSkul0tHjx7VFVdccTG/n99FR0drX7NLbVffFuhSehX50VZFR0cHugwANpKdna0lS5aoqKhII0eO1AMPPKAtW7ZIku644w7l5OSorKxM//Ef/6GtW7eqvLxczc3N+t73vqeoqChNnDhRCQkJ5903bNgwr2swDIs333xTzz//vCRp69atevzxx5WcnKyMjIw+/toAgJ5cccUV2rx583n3jxw5Ups2bep231NPPeX597Zt2zz/TktLkyR997vfPa+d3NzcPtdmuBvq1Vdf1ebNmzV8+HBJ0qOPPqqioqI+dwgAsB/DsBgwYIAiIiIUEhIiSYqI8P7gbFtbm9LS0rR582YdOXJEs2fPlsPh0IMPPiiXy9X3qgEAfmUYFtddd50effRR1dfX61e/+pW+//3va9KkSV41/vzzz3v2ieXl5cnhcKioqEiJiYkqKSm5tMoBAH5jGBYPPfSQZs6cqe9973saOHCgcnJy9NBDDxk2vHfvXu3Zs0c33XSTJKmqqsqzLy0tLU2VlZWXVjkAwG8MD3A/99xznn+f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\n",
      "text/plain": [
       "<Figure size 401.625x360 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.factorplot(x = \"Sex\", y = \"Age\", hue = \"Pclass\",data = train_df, kind = \"box\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "execution_count": null,
   "metadata": {
    "papermill": {
     "duration": 0.047906,
     "end_time": "2020-09-08T17:54:41.291739",
     "exception": false,
     "start_time": "2020-09-08T17:54:41.243833",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "sınıf olarak pclassı ekleyelim.y ekseni yaş x ekseni cinsiyet üç tane boxımız var yaş medyanlaru birinci sınıftaki yaş ortalaması 40,ikincisi sınıfın yaklaşık 30 üçüncü sınıfın ise 25.yani en yaşlılar 1de en gençler 3.sınıfda.jamese baktığımızda yukardan 3.sınıfta diyebilriz.pclasss benim jamesin yaşını anlamam için işe yarar"
   ]
  },
  {
   "cell_type": "markdown",
   "execution_count": null,
   "metadata": {
    "papermill": {
     "duration": 0.046163,
     "end_time": "2020-09-08T17:54:41.384391",
     "exception": false,
     "start_time": "2020-09-08T17:54:41.338228",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "1st class passengers are older than 2nd, and 2nd is older than 3rd class.\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2020-09-08T17:54:41.607574Z",
     "iopub.status.busy": "2020-09-08T17:54:41.492708Z",
     "iopub.status.idle": "2020-09-08T17:54:42.188204Z",
     "shell.execute_reply": "2020-09-08T17:54:42.187527Z"
    },
    "papermill": {
     "duration": 0.757504,
     "end_time": "2020-09-08T17:54:42.188324",
     "exception": false,
     "start_time": "2020-09-08T17:54:41.430820",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "image/png": "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YQ0YMCCULQCAbYQ0gK+88koVFRVJkj7//HMlJCSE3fwvAJyrkE5BXHbZZbrwwguVmZmpiIgIPf7446EsDwC2EuH/4cSsYdu3b9fll19uug0AsBxXwgGAIQQwABhCAAOAIQQwABhCAAOAIQQwABhCAAOAISFfD/hcbN++3XQLABBUbV3fYLsLMQDgfMEUBAAYQgADgCEEMAAYErYB/NRTTykjI0OZmZnasWOH6XZa2LVrl6677jq9+uqrpltpITs7WxkZGfrlL3+p4uJi0+00q6+v17333qtp06bplltu0aZNm0y31IrP51NKSorefvtt061Iknbu3Knk5GRlZWUpKytLTzzxhOmWWli/fr1+8Ytf6Oabb9b7779vup1mTU1NevTRR5WZmamsrCzt3r3b0nq2PAuiq+x880+v16snnnhC48ePN91KC1u2bFFZWZny8vJUW1urm266SZMnTzbdliRp06ZNuuiii3TnnXfqwIEDuuOOO3TNNdeYbquFlStXql+/fqbbaOb1ejVlyhQtWLDAdCut1NbWasWKFXrrrbfk9Xq1bNkyXXXVVabbkiT985//1NGjR/XGG2/om2++0ZNPPqkXXnjBsnphGcB2vvmnw+HQ6tWrtXr1atOttHDFFVfokksukST17dtX9fX1amxsVI8ePQx3dvpOzGccPHhQAwcONNhNa7t371Z5ebmuvvpq0600O378uOkWAiopKdH48eMVGxur2NhYWx2d7927t/n7YMSIEaqoqLD0+yAspyCqq6vVv3//5sdnbv5pB1FRUYqOjjbdRis9evSQ0+mUJL355ptKTk62Rfh+X2Zmph588EHNnz/fdCstLFmyRA899JDpNlrwer3avn27ZsyYoVtvvVVbtmwx3VKz/fv3y+/367777pPb7VZJSYnplpqNGTNGH330kRobG/XVV19p3759qq2ttaxeWB4Bn+vNP9Hae++9p/z8fL388sumW2nljTfeUGlpqebMmaP169fb4nO6bt06XXrppRo+fLjpVloYO3asZs6cqZSUFO3Zs0e33367iouL5XA4TLcmSaqsrNTy5ctVUVGh2267TZs2bbLF5/Oqq67Sp59+qltvvVU/+9nP9JOf/KRVngRTWAYwN//snA8//FCrVq3Siy++qD59+phup9nOnTsVHx+vwYMHKzExUY2NjfJ4PIqPjzfdmjZv3qx9+/Zp8+bNOnTokBwOhwYNGqQJEyYY7WvUqFEaNWqUJGnkyJEaMGCAKisrbfGDIj4+XuPGjVNUVJRGjBih3r172+bzKUn3339/8/+vu+46S/sKyykIbv7ZcUePHlV2drZeeOEFW/0xSZK2bdvWfEReXV0tr9fbYorJpOeee05vvfWW/vrXv+qWW27RPffcYzx8JSk/P19//vOfJUlVVVWqqamxzdz5xIkTtWXLFjU1Ncnj8djq8/nFF1/o4YcfliR98MEH+vnPf67ISOtiMiyPgO1888+dO3dqyZIlOnDggKKiolRUVKRly5YZD2Pf5SsAAAKASURBVL2CggLV1tbqvvvua962ZMkSDRkyxGBXp2VmZmrBggVyu93y+Xx67LHHLP2mCAeTJk3Sgw8+qKKiIp08eVILFy60zfTDwIEDNWXKFE2fPl319fV65JFHbPP5HDNmjPx+vzIyMtSnTx8tWbLE0nqsBQEAhtjjxw4AnIcIYAAwhAAGAEMIYAAwhAAGAEMIYISN/fv3a9y4ccrKytK0adP0q1/9Shs3buz0eFlZWdq1a1cQOwRaCsvzgHH+GjlypNauXStJqqur00033aSkpCRbrr8BEMAIW/369ZPL5dLevXu1aNEiRUVFKTIyUs8//7yOHTumOXPmyOl0atq0aXI4HPrjH/+oHj16KD09Xb/+9a8lSYWFhXryySdVV1enlStX2uLCFIQPpiAQtvbv36+6ujrV1NTo0Ucf1dq1a3XZZZfpH//4hySptLRUzzzzjK6++motWrRIq1ev1uuvv66SkhL5fD5Jp9ctyM3NVXJysq0WqUd44AgYYWXPnj3KysqS3+9Xr169tGTJEsXExOiZZ56Rz+fT4cOHdcMNN0iShg8frv79+6umpka9evVSXFycJLVYgPvMrcQHDhyourq60L8ghDUCGGHl+3PAZ2RlZenOO+9UcnKyXnrpJXm9XklSz549JUmRkZFqampqc7zvr4nMVfsINqYgEPbq6uo0YsQInTx5Uu+//74aGhpaPN+/f381NjaqsrJSfr9fd911l44cOWKoW5xPOAJG2Js2bZpmzpyp4cOHN9+g8vu3OZKkxx9/XLNnz5YkpaWl6Uc/+pGJVnGeYTU0ADCEKQgAMIQABgBDCGAAMIQABgBDCGAAMIQABgBDCGAAMOT/AfETIMRyisfGAAAAAElFTkSuQmCC\n",
      "text/plain": [
       "<Figure size 360x360 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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rV4APePXAK6pqdFFF13kOY6NjVV1dfWPAri8vLzbbZ533nm66667fFbjmfTdE8FWLwDvJSUltXverwH8n7MdbrdbYWFhP7quo2IBIJT49VHkIUOGqKamxnN8+PBhxcXF+bMEAAgYfg3gq666SsXFxZKkzz77TAkJCZbP/wJAoPLrFMTll1+uiy66SBkZGQoLC9Pvfvc7f3YPAAHFr7ehdceZ3oYGAMGK5SgBwBACGAAMIYABwBACGAAMIYABwBACGAAMIYABwJCAXJC9rKzMdAkA4FPtPd8QcA9iAEBvwRQEABhCAAOAIQQwABjSKwL4qaee0owZM5SRkaFdu3aZLqdLe/fu1Q033KB169aZLqVbli5dqhkzZujnP/+5SkpKTJfTqcbGRmVnZ2vmzJmaPn26tm7darqkbnG5XLLb7dq4caPpUrq0e/duTZw4UbNmzdKsWbP0xBNPmC6pUw0NDZo7d65mzZqljIwMbd++3W99B+RdEL700Ucf6euvv9aGDRs63Ag0kDidTj3xxBMaP3686VK65cMPP9S+ffu0YcMGHTlyRNOmTdOkSZNMl9WhrVu3auzYsZo9e7YOHDigO+64Q9ddd53psrr00ksvadCgQabL6Ban06nJkydr4cKFpkvplnfeeUcjR47UAw88oKqqKmVmZqqoqMgvfYd8AJvaCLSnoqKitHLlSq1cudJ0Kd1y5ZVXaty4cZKkgQMHqrGxUS0tLerTp4/hytqXnp7u+X1lZaWGDBlisJru+eKLL1RRUaFrr73WdCnd0tDQYLqEMzJ48GB9/vnnkqSjR49q8ODBfus75AO4uxuBBoqIiAhFRATPX0ufPn1ks9kkSW+99ZYmTpwYsOH77zIyMnTo0CGtWLHCdCldWrJkiR599FHl5+ebLqVbnE6nysrKdOedd6qxsVFZWVn66U9/arqsDt14443auHGjUlJSdPToUb388st+6zt4/qf3UHc3AoV3tmzZory8PL3yyiumS+mWN954Q+Xl5Zo3b542bdoUsP8m8vPzdemll+rcc881XUq3jRkzRnfffbfsdru++uor3X777SopKVFUVJTp0tr17rvvavjw4Vq9erX27NmjhQsX6u233/ZL3yEfwGwEar3t27drxYoVWrVqlc466yzT5XRq9+7dio2N1bBhw5SUlKSWlhbV1dUpNjbWdGnt2rZtm7799ltt27ZNhw4dUlRUlIYOHaoJEyaYLq1Do0aN0qhRoyRJI0eOVFxcnKqqqgL2m8jHH3+sq6++WtKpbx5VVVU6efKkX34SDfm7INgI1FrHjh3T0qVL9fLLLwfFh0Q7d+70jNJramrkdDr9Oud3pp577jm9/fbbevPNNzV9+nTNmTMnoMNXkvLy8vTaa69Jkqqrq1VbWxvQc+3nnXeePv30U0nSgQMH1L9/f79NA/aKR5GfeeYZ7dy507MR6JgxY0yX1KHdu3dryZIlOnDggCIiIjRkyBAtX748YMNtw4YNWr58uUaOHOk5t2TJEg0fPgu0DHgAAAM4SURBVNxgVR1zuVxauHChKisr5XK5NHfuXF1//fWmy+qW5cuXKzExUbfccovpUjr1/fff68EHH5TT6VRTU5Pmzp2ra665xnRZHWpoaNAjjzyi2tpanTx5UtnZ2X67C6lXBDAABKKQn4IAgEBFAAOAIQQwABhCAAOAIQQwABgS8g9ioPd4/fXX9e6776pv375qbGzU/fffr61bt+q2225Tfn6+Bg8erJkzZ7Z5zeeff64nn3xSra2tcjqdGj9+vB588MGAfTIOoYUARkj47rvv9OabbyovL0+RkZHav3+/Fi1a1OWSnosXL9a8efM0btw4tba26u6779a//vUvjR071k+VozdjCgIh4fjx4zpx4oSam5slSeeff77WrVunWbNmae/evZKkf/7zn7rrrrt000036a9//aukU0/yHT9+XJIUHh6ul156SWPHjtXGjRt13333afbs2ZoyZYrf1gZA78IIGCFhzJgxGjdunOx2u6655hpNnDjxR+sS19bWatWqVdq7d68eeughTZw4UXPnzlV2drYuvvhiXXXVVZoyZYoSEhIkSRUVFXrnnXd09OhR3XzzzZo2bZrCwxmzwHf414SQsXTpUq1bt05jxozRqlWrdPvtt7dZDe8nP/mJJOnCCy9UZWWlJOmGG27Qn//8Z916663as2ePbrrpJu3Zs0fSqbWOIyIiFBMTo4EDB+rIkSP+f1MIaYyAERLcbreampo8K3HNmjVLaWlpOnnypOea9j5Yc7lcOvvss5Wenq709HS98MIL2rJli4YPH67W1tY27fPBHHyNETBCQl5enh599FHPiPfYsWNqbW1ts8xkWVmZJGnPnj1KTEzU8ePHlZaWpurqas81hw4d0jnnnCNJ+uSTTzzLVTY0NATsgkgIXoyAERJuueUWffnll5o+fbpsNpuam5u1aNEirV692nNNbGys7rrrLn377bdauHChBgwYoMcee0xZWVmKjIxUc3OzLrnkEv3sZz9Tfn6+EhMTlZ2dra+//lr33nsv87/wOVZDA9qxceNG7du3TwsWLDBdCkIY39IBwBBGwABgCCNgADCEAAYAQwhgADCEAAYAQwhgADDk/wNy1LIfs/nfGwAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<Figure size 360x360 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.factorplot(x = \"Parch\", y = \"Age\", data = train_df, kind = \"box\")\n",
    "sns.factorplot(x = \"SibSp\", y = \"Age\", data = train_df, kind = \"box\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "execution_count": null,
   "metadata": {
    "papermill": {
     "duration": 0.04689,
     "end_time": "2020-09-08T17:54:42.282775",
     "exception": false,
     "start_time": "2020-09-08T17:54:42.235885",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "parch 0 1 2 olanlar için yaş ortalaması hemen hemen aynı ama 3 4 5 e baktığımızda ortalamanın yüksekte olduğunu düşünebilriz.jamese baktığımızda parchsi0 yani james 20 yaşında diyebilriz.\n",
    "sibsp için 0 1 2 için otalam aynı ama 3ten sonrası için yaş ortalaması 10 fln jamese baktığımızda burdada 25 yaşında çıkıyor."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2020-09-08T17:54:42.382634Z",
     "iopub.status.busy": "2020-09-08T17:54:42.381697Z",
     "iopub.status.idle": "2020-09-08T17:54:42.384789Z",
     "shell.execute_reply": "2020-09-08T17:54:42.384215Z"
    },
    "papermill": {
     "duration": 0.054844,
     "end_time": "2020-09-08T17:54:42.384917",
     "exception": false,
     "start_time": "2020-09-08T17:54:42.330073",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "#train_df[\"Sex\"] = [1 if i == \"male\" else 0 for i in train_df[\"Sex\"]]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2020-09-08T17:54:42.487200Z",
     "iopub.status.busy": "2020-09-08T17:54:42.486401Z",
     "iopub.status.idle": "2020-09-08T17:54:42.706823Z",
     "shell.execute_reply": "2020-09-08T17:54:42.707383Z"
    },
    "papermill": {
     "duration": 0.275453,
     "end_time": "2020-09-08T17:54:42.707549",
     "exception": false,
     "start_time": "2020-09-08T17:54:42.432096",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.heatmap(train_df[[\"Age\",\"Sex\",\"SibSp\",\"Parch\",\"Pclass\"]].corr(), annot = True)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "execution_count": null,
   "metadata": {
    "papermill": {
     "duration": 0.047739,
     "end_time": "2020-09-08T17:54:42.803132",
     "exception": false,
     "start_time": "2020-09-08T17:54:42.755393",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "heatmape baktığımızda dediğimiz gibi <age ve cinsiyet arasıbda bir bağ yok."
   ]
  },
  {
   "cell_type": "markdown",
   "execution_count": null,
   "metadata": {
    "papermill": {
     "duration": 0.047528,
     "end_time": "2020-09-08T17:54:42.898753",
     "exception": false,
     "start_time": "2020-09-08T17:54:42.851225",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "Age is not correlated with sex but it is correlated with parch, sibsp and pclass."
   ]
  },
  {
   "cell_type": "markdown",
   "execution_count": null,
   "metadata": {
    "papermill": {
     "duration": 0.047932,
     "end_time": "2020-09-08T17:54:42.994723",
     "exception": false,
     "start_time": "2020-09-08T17:54:42.946791",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "burda age featurune alıp agelaerin  boş olanını bulup bunların indeksini bul ve indexleri dolaş ve sibsp, parch ve pclassa bakarak agein predicitionunu al\n",
    "ve bunalrın medyanını alıcam ve bazı yerlerde prediction yapamayız bunun için bazı yerlerde trainin dataframini kullanıcaz.eğer boş değilse i. indeksi age pred olcak \n",
    "ama eğer boşsa i. indelsi age med olcak.burda train df deki bütün null değerleri ortadan kaldırdık."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2020-09-08T17:54:43.100414Z",
     "iopub.status.busy": "2020-09-08T17:54:43.099463Z",
     "iopub.status.idle": "2020-09-08T17:54:43.104032Z",
     "shell.execute_reply": "2020-09-08T17:54:43.103397Z"
    },
    "papermill": {
     "duration": 0.06164,
     "end_time": "2020-09-08T17:54:43.104155",
     "exception": false,
     "start_time": "2020-09-08T17:54:43.042515",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "ename": "SyntaxError",
     "evalue": "invalid syntax (<ipython-input-43-1e05850b7ff6>, line 4)",
     "output_type": "error",
     "traceback": [
      "\u001b[0;36m  File \u001b[0;32m\"<ipython-input-43-1e05850b7ff6>\"\u001b[0;36m, line \u001b[0;32m4\u001b[0m\n\u001b[0;31m    \"SibSp\"]) &(train_df[\"Parch\"] == train_df.iloc[i]\u001b[0m\n\u001b[0m          ^\u001b[0m\n\u001b[0;31mSyntaxError\u001b[0m\u001b[0;31m:\u001b[0m invalid syntax\n"
     ]
    }
   ],
   "source": [
    "index_nan_age = list(train_df[\"Age\"][train_df[\"Age\"].isnull()].index)\n",
    "for i in index_nan_age:\n",
    "    age_pred = train_df[\"Age\"][((train_df[\"SibSp\"] == train_df.iloc[i]\n",
    "                    \"SibSp\"]) &(train_df[\"Parch\"] == train_df.iloc[i]\n",
    "                [\"Parch\"])& (train_df[\"Pclass\"] == train_df.iloc[i]\n",
    "                                              [\"Pclass\"]))].median()\n",
    "    age_med = train_df[\"Age\"].median()\n",
    "    if not np.isnan(age_pred):\n",
    "        train_df[\"Age\"].iloc[i] = age_pred\n",
    "    else:\n",
    "        train_df[\"Age\"].iloc[i] = age_med"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2020-09-08T17:54:43.234458Z",
     "iopub.status.busy": "2020-09-08T17:54:43.233615Z",
     "iopub.status.idle": "2020-09-08T17:54:43.238353Z",
     "shell.execute_reply": "2020-09-08T17:54:43.238904Z"
    },
    "papermill": {
     "duration": 0.086176,
     "end_time": "2020-09-08T17:54:43.239074",
     "exception": false,
     "start_time": "2020-09-08T17:54:43.152898",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Name</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>Embarked</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>6</td>\n",
       "      <td>0.0</td>\n",
       "      <td>3</td>\n",
       "      <td>Moran, Mr. James</td>\n",
       "      <td>male</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>330877</td>\n",
       "      <td>8.4583</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Q</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>17</th>\n",
       "      <td>18</td>\n",
       "      <td>1.0</td>\n",
       "      <td>2</td>\n",
       "      <td>Williams, Mr. Charles Eugene</td>\n",
       "      <td>male</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>244373</td>\n",
       "      <td>13.0000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>19</th>\n",
       "      <td>20</td>\n",
       "      <td>1.0</td>\n",
       "      <td>3</td>\n",
       "      <td>Masselmani, Mrs. Fatima</td>\n",
       "      <td>female</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>2649</td>\n",
       "      <td>7.2250</td>\n",
       "      <td>NaN</td>\n",
       "      <td>C</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>26</th>\n",
       "      <td>27</td>\n",
       "      <td>0.0</td>\n",
       "      <td>3</td>\n",
       "      <td>Emir, Mr. Farred Chehab</td>\n",
       "      <td>male</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>2631</td>\n",
       "      <td>7.2250</td>\n",
       "      <td>NaN</td>\n",
       "      <td>C</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>27</th>\n",
       "      <td>29</td>\n",
       "      <td>1.0</td>\n",
       "      <td>3</td>\n",
       "      <td>O'Dwyer, Miss. Ellen \"Nellie\"</td>\n",
       "      <td>female</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>330959</td>\n",
       "      <td>7.8792</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Q</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1282</th>\n",
       "      <td>1300</td>\n",
       "      <td>NaN</td>\n",
       "      <td>3</td>\n",
       "      <td>Riordan, Miss. Johanna Hannah\"\"</td>\n",
       "      <td>female</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>334915</td>\n",
       "      <td>7.7208</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Q</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1284</th>\n",
       "      <td>1302</td>\n",
       "      <td>NaN</td>\n",
       "      <td>3</td>\n",
       "      <td>Naughton, Miss. Hannah</td>\n",
       "      <td>female</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>365237</td>\n",
       "      <td>7.7500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Q</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1287</th>\n",
       "      <td>1305</td>\n",
       "      <td>NaN</td>\n",
       "      <td>3</td>\n",
       "      <td>Spector, Mr. Woolf</td>\n",
       "      <td>male</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>A.5. 3236</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1290</th>\n",
       "      <td>1308</td>\n",
       "      <td>NaN</td>\n",
       "      <td>3</td>\n",
       "      <td>Ware, Mr. Frederick</td>\n",
       "      <td>male</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>359309</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1291</th>\n",
       "      <td>1309</td>\n",
       "      <td>NaN</td>\n",
       "      <td>3</td>\n",
       "      <td>Peter, Master. Michael J</td>\n",
       "      <td>male</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2668</td>\n",
       "      <td>22.3583</td>\n",
       "      <td>NaN</td>\n",
       "      <td>C</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>256 rows × 12 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "      PassengerId  Survived  Pclass                             Name     Sex  \\\n",
       "5               6       0.0       3                 Moran, Mr. James    male   \n",
       "17             18       1.0       2     Williams, Mr. Charles Eugene    male   \n",
       "19             20       1.0       3          Masselmani, Mrs. Fatima  female   \n",
       "26             27       0.0       3          Emir, Mr. Farred Chehab    male   \n",
       "27             29       1.0       3    O'Dwyer, Miss. Ellen \"Nellie\"  female   \n",
       "...           ...       ...     ...                              ...     ...   \n",
       "1282         1300       NaN       3  Riordan, Miss. Johanna Hannah\"\"  female   \n",
       "1284         1302       NaN       3           Naughton, Miss. Hannah  female   \n",
       "1287         1305       NaN       3               Spector, Mr. Woolf    male   \n",
       "1290         1308       NaN       3              Ware, Mr. Frederick    male   \n",
       "1291         1309       NaN       3         Peter, Master. Michael J    male   \n",
       "\n",
       "      Age  SibSp  Parch     Ticket     Fare Cabin Embarked  \n",
       "5     NaN      0      0     330877   8.4583   NaN        Q  \n",
       "17    NaN      0      0     244373  13.0000   NaN        S  \n",
       "19    NaN      0      0       2649   7.2250   NaN        C  \n",
       "26    NaN      0      0       2631   7.2250   NaN        C  \n",
       "27    NaN      0      0     330959   7.8792   NaN        Q  \n",
       "...   ...    ...    ...        ...      ...   ...      ...  \n",
       "1282  NaN      0      0     334915   7.7208   NaN        Q  \n",
       "1284  NaN      0      0     365237   7.7500   NaN        Q  \n",
       "1287  NaN      0      0  A.5. 3236   8.0500   NaN        S  \n",
       "1290  NaN      0      0     359309   8.0500   NaN        S  \n",
       "1291  NaN      1      1       2668  22.3583   NaN        C  \n",
       "\n",
       "[256 rows x 12 columns]"
      ]
     },
     "execution_count": 44,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_df[train_df[\"Age\"].isnull()]\n"
   ]
  },
  {
   "cell_type": "markdown",
   "execution_count": null,
   "metadata": {
    "papermill": {
     "duration": 0.048365,
     "end_time": "2020-09-08T17:54:43.336062",
     "exception": false,
     "start_time": "2020-09-08T17:54:43.287697",
     "status": "completed"
    },
    "tags": []
   },
   "source": []
  },
  {
   "cell_type": "markdown",
   "execution_count": null,
   "metadata": {
    "papermill": {
     "duration": 0.048176,
     "end_time": "2020-09-08T17:54:43.433091",
     "exception": false,
     "start_time": "2020-09-08T17:54:43.384915",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "\n",
    "Feature Engineering\n",
    "yani burda farklı featurları kullanarak yeni featurelar elde edeceğiz."
   ]
  },
  {
   "cell_type": "markdown",
   "execution_count": null,
   "metadata": {
    "papermill": {
     "duration": 0.048332,
     "end_time": "2020-09-08T17:54:43.529789",
     "exception": false,
     "start_time": "2020-09-08T17:54:43.481457",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "Name -- Title :name featurenin içindeki özellşklerş kullanarak neler yapıcaz bunu inceleyeceğiz"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2020-09-08T17:54:43.671778Z",
     "iopub.status.busy": "2020-09-08T17:54:43.670846Z",
     "iopub.status.idle": "2020-09-08T17:54:43.675619Z",
     "shell.execute_reply": "2020-09-08T17:54:43.674982Z"
    },
    "papermill": {
     "duration": 0.097471,
     "end_time": "2020-09-08T17:54:43.675756",
     "exception": false,
     "start_time": "2020-09-08T17:54:43.578285",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0                              Braund, Mr. Owen Harris\n",
       "1    Cumings, Mrs. John Bradley (Florence Briggs Th...\n",
       "2                               Heikkinen, Miss. Laina\n",
       "3         Futrelle, Mrs. Jacques Heath (Lily May Peel)\n",
       "4                             Allen, Mr. William Henry\n",
       "5                                     Moran, Mr. James\n",
       "6                              McCarthy, Mr. Timothy J\n",
       "7                       Palsson, Master. Gosta Leonard\n",
       "8    Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)\n",
       "9                  Nasser, Mrs. Nicholas (Adele Achem)\n",
       "Name: Name, dtype: object"
      ]
     },
     "execution_count": 45,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_df[\"Name\"].head(10)"
   ]
  },
  {
   "cell_type": "markdown",
   "execution_count": null,
   "metadata": {
    "papermill": {
     "duration": 0.048878,
     "end_time": "2020-09-08T17:54:43.774965",
     "exception": false,
     "start_time": "2020-09-08T17:54:43.726087",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "burda bana train dataframeinin içindeki ilk 10 yolcuyu yazdırcak\n",
    "burda insanların isimleri ve ünvanları yani titleları var.\n",
    "ilk önce bu namein içindeki titleları çekmek isityorum. noktaya vw virgüle göre ayırabilrim\n",
    "önemli titlelar iste hayatta kalma olasılıkları daha yüksek diyebiliriz"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2020-09-08T17:54:43.882169Z",
     "iopub.status.busy": "2020-09-08T17:54:43.881211Z",
     "iopub.status.idle": "2020-09-08T17:54:43.884369Z",
     "shell.execute_reply": "2020-09-08T17:54:43.883765Z"
    },
    "papermill": {
     "duration": 0.060053,
     "end_time": "2020-09-08T17:54:43.884491",
     "exception": false,
     "start_time": "2020-09-08T17:54:43.824438",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "name = train_df[\"Name\"]\n",
    "train_df[\"Title\"] = [i.split(\".\")[0].split(\",\")[-1].strip() for i in name]\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2020-09-08T17:54:43.990074Z",
     "iopub.status.busy": "2020-09-08T17:54:43.989265Z",
     "iopub.status.idle": "2020-09-08T17:54:43.993287Z",
     "shell.execute_reply": "2020-09-08T17:54:43.992547Z"
    },
    "papermill": {
     "duration": 0.060235,
     "end_time": "2020-09-08T17:54:43.993410",
     "exception": false,
     "start_time": "2020-09-08T17:54:43.933175",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0        Mr\n",
       "1       Mrs\n",
       "2      Miss\n",
       "3       Mrs\n",
       "4        Mr\n",
       "5        Mr\n",
       "6        Mr\n",
       "7    Master\n",
       "8       Mrs\n",
       "9       Mrs\n",
       "Name: Title, dtype: object"
      ]
     },
     "execution_count": 47,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_df[\"Title\"].head(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2020-09-08T17:54:44.114841Z",
     "iopub.status.busy": "2020-09-08T17:54:44.113739Z",
     "iopub.status.idle": "2020-09-08T17:54:44.340225Z",
     "shell.execute_reply": "2020-09-08T17:54:44.339477Z"
    },
    "papermill": {
     "duration": 0.297559,
     "end_time": "2020-09-08T17:54:44.340346",
     "exception": false,
     "start_time": "2020-09-08T17:54:44.042787",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.countplot(x=\"Title\", data = train_df)\n",
    "plt.xticks(rotation = 60)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "execution_count": null,
   "metadata": {
    "papermill": {
     "duration": 0.04893,
     "end_time": "2020-09-08T17:54:44.438842",
     "exception": false,
     "start_time": "2020-09-08T17:54:44.389912",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "biz burdaki titleları gruplayabiliriz.kadınların kullandığı master msr ve diğerleri şklinde kategori yapabiliriz"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2020-09-08T17:54:44.552756Z",
     "iopub.status.busy": "2020-09-08T17:54:44.551652Z",
     "iopub.status.idle": "2020-09-08T17:54:44.556841Z",
     "shell.execute_reply": "2020-09-08T17:54:44.556099Z"
    },
    "papermill": {
     "duration": 0.068649,
     "end_time": "2020-09-08T17:54:44.556965",
     "exception": false,
     "start_time": "2020-09-08T17:54:44.488316",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0     2\n",
       "1     1\n",
       "2     1\n",
       "3     1\n",
       "4     2\n",
       "5     2\n",
       "6     2\n",
       "7     0\n",
       "8     1\n",
       "9     1\n",
       "10    1\n",
       "11    1\n",
       "12    2\n",
       "13    2\n",
       "14    1\n",
       "15    1\n",
       "16    0\n",
       "17    2\n",
       "18    1\n",
       "19    1\n",
       "Name: Title, dtype: int64"
      ]
     },
     "execution_count": 49,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#kategorik hale getirme\n",
    "train_df[\"Title\"] = train_df[\"Title\"].replace([\"Lady\", \"the Countess\", \"Capt\", \"Col\",\"Don\", \"Major\",\"Rev\",\"Sir\", \"Jonkhear\", \"Dona\"],\"other\")\n",
    "train_df[\"Title\"] = [0 if i == \"Master\" else 1 if i == \"Miss\" or i == \"Ms\" or i == \"Mlle\" or i == \"Mrs\" else 2 if i == \"Mr\" else 3 for i in train_df[\"Title\"]]\n",
    "train_df[\"Title\"].head(20)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2020-09-08T17:54:44.670626Z",
     "iopub.status.busy": "2020-09-08T17:54:44.664943Z",
     "iopub.status.idle": "2020-09-08T17:54:44.801474Z",
     "shell.execute_reply": "2020-09-08T17:54:44.800842Z"
    },
    "papermill": {
     "duration": 0.194742,
     "end_time": "2020-09-08T17:54:44.801612",
     "exception": false,
     "start_time": "2020-09-08T17:54:44.606870",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.countplot(x=\"Title\", data = train_df)\n",
    "plt.xticks(rotation = 60)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2020-09-08T17:54:44.910794Z",
     "iopub.status.busy": "2020-09-08T17:54:44.909699Z",
     "iopub.status.idle": "2020-09-08T17:54:45.246187Z",
     "shell.execute_reply": "2020-09-08T17:54:45.245561Z"
    },
    "papermill": {
     "duration": 0.395042,
     "end_time": "2020-09-08T17:54:45.246316",
     "exception": false,
     "start_time": "2020-09-08T17:54:44.851274",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "image/png": "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6i9/+Yt/uaamRl988cVJ7WPPnj3asGFDR0eLOB0xll0Z49daS0uLfvSjH2n69Ol6/vnn9cILL6h///564IEHVF5e3ubirkhk21kQkeLiiy/W22+/rebmZkVHR2vDhg1KT0/Xl19+qZdfflmrV69WVFSULrzwQi1atEifffaZ7rvvPkVFRam5uVm/+tWvVFhYqA8++ECPP/64brnlFuXn5+vgwYNqbm7WT3/6UyUnJ+uqq67SmDFjFB8fr5kzZ5p+27Y43lge+/4vuugi/frXv1b37t0VHx+vJUuWqFu3bqbjG3ei49dZPz/f9Oabb+q8887TqFGj/OtuvfVWXXnlldq4caOcTqfOOeccSdI777yjZ599Vvv27dOSJUs0dOhQrVmzRi+//LKioqKUkZGh2267TcuXL9fu3bu1Z88erV69WtHR0abeniTuBaFu3brp4osv1jvvvCNJ+tvf/qaxY8dKkg4fPqzf//73WrdunT7++GP985//VGlpqS6//HKtXr1aP/nJT+R2u3X77bfr0ksv1ezZs/XMM89o9OjReuaZZ/R///d/evjhhyVJTU1NGjNmTKf+w3O8sTz2/T/77LO6//779eyzz+raa69VXV2dydhh40THr6v4+OOPNXTo0FbrLMvS0KFDlZaWptzcXF155ZX+9U899ZRyc3P1wgsvaPfu3XrllVe0du1arVmzRhs3btRnn30mSWpsbNQf//hH4+UrcQQsSZowYYL+8pe/qE+fPnI6nTrrrLMkST179tTdd98tSfroo49UV1en9PR0zZ49W1988YWysrI0bNgw/x8YSdq2bZsOHDigl156SZJ05MgR/7bU1NQQviszAo2l9L/3P2HCBBUUFGjixIm69tprOQXxGCcyfl1Jc3Nzm3U+n09RUa2PHY9eM+B0OlVZWakPP/xQNTU1ys3NlSQdOnRIe/fulRRe40gBS7r88stVWFioPn36KCsrS9LXf0sWFhbqz3/+s/r06aM777xTkjRkyBD9+c9/1ltvvaVHH31U119/vf+fQdLXRzEPPPCAhg0b1uZ1usI/s9sby6OOvv/Jkydr9OjRevXVVzVz5kwtXbpUF1xwgYm4YedExq+ruOCCC7R27dpW63w+n6qrqzV69OhW6489mvX5fOrWrZvGjRunwsLCVs8rLy8Pq3Hs8lMQ0tcf7EsuuUTPP/+8xo8fL+nrvzGjo6PVp08f7du3T9u3b1djY6M2bNigXbt2KSMjQ3PnztX27dsVFRWlhoYGSV/P47366quSpOrqaj399NPG3pcJ7Y3lN61YsUIxMTHKzs7WNddco48++ijEKcPXiYxfV5Genq49e/Zo06ZN/nUrV67UiBEj1KtXL/+fufa4XC698847OnLkiHw+nxYvXqwvv/wyFLFPCkfA/zVhwgQdOHBAcXFxkqRevXopPT1d119/vZKTk3XHHXfoF7/4hYqKilRYWKizzjpL0dHR+ulPf6revXvrH//4h4qKijRnzhwtXLhQN910k1paWnc7EpgAAAHKSURBVPSTn/zE8DsLvW+O5Tf169dPt956q771rW/pW9/6lm699dYQJwxvwcavq4iKitJTTz2lgoICLV26VD6fTyNGjFBBQYG2bNmiBQsW+K+u/aZ+/fopNzdXU6dOVXR0tDIyMtS9e/cQv4PgbL0QAwAQGFMQAGAIBQwAhlDAAGAIBQwAhlDAAGAIp6Gh03rooYe0Y8cOud1uHTlyRIMGDdKWLVu0du1aDRs2TKWlpcrKytLy5cvVu3dv3XzzzaYjo4uhgNFp3X///ZKkkpIS7dq1SwsWLPBvO3oHu29ebQaEEgWMLuX+++9XVlaW1q5d67+D3bEee+wxvfvuu2pubtbNN9+s6667zlBSdAXMAaNLOvYOdke9++672rt3r9asWaNVq1bpiSeeCMvLV9F5cAQM/Nd7772nyspKTZs2TdLXNwR3u90aOHCg4WTorChg4L9iY2N1ww03+O98B9iNKQh0Scfewe6o1NRUvfbaa2ppadFXX32lRYsWGUqHroIjYHRJF1xwgf8OdkfvOpaWlqaRI0cqOztbPp9PN910k+GU6Oy4GxoAGMIUBAAYQgEDgCEUMAAYQgEDgCEUMAAYQgEDgCEUMAAY8v/Va/8ODulO8gAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<Figure size 360x360 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "g = sns.factorplot(x = \"Title\", y = \"Survived\", data = train_df, kind = \"bar\")\n",
    "g.set_xticklabels([\"Master\",\"Mrs\",\"Mr\",\"Other\"])\n",
    "g.set_ylabels(\"Survival Probability\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "execution_count": null,
   "metadata": {
    "papermill": {
     "duration": 0.049582,
     "end_time": "2020-09-08T17:54:45.345960",
     "exception": false,
     "start_time": "2020-09-08T17:54:45.296378",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "çıkan plotta hayatta kalma oranı en yüksek kaadınlae en düşük erkekler.artık train df içinden name featurunden çıkarıyorum yani ben name featuremı train dfmi eğitirken kullanmayacağım.artık name yerine title var."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2020-09-08T17:54:45.453675Z",
     "iopub.status.busy": "2020-09-08T17:54:45.452571Z",
     "iopub.status.idle": "2020-09-08T17:54:45.456088Z",
     "shell.execute_reply": "2020-09-08T17:54:45.455344Z"
    },
    "papermill": {
     "duration": 0.060416,
     "end_time": "2020-09-08T17:54:45.456210",
     "exception": false,
     "start_time": "2020-09-08T17:54:45.395794",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "train_df.drop(labels = [\"Name\"], axis = 1, inplace = True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2020-09-08T17:54:45.574230Z",
     "iopub.status.busy": "2020-09-08T17:54:45.573470Z",
     "iopub.status.idle": "2020-09-08T17:54:45.578787Z",
     "shell.execute_reply": "2020-09-08T17:54:45.578029Z"
    },
    "papermill": {
     "duration": 0.072722,
     "end_time": "2020-09-08T17:54:45.578911",
     "exception": false,
     "start_time": "2020-09-08T17:54:45.506189",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>Embarked</th>\n",
       "      <th>Title</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>0.0</td>\n",
       "      <td>3</td>\n",
       "      <td>male</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>A/5 21171</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1</td>\n",
       "      <td>female</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>PC 17599</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>C85</td>\n",
       "      <td>C</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>1.0</td>\n",
       "      <td>3</td>\n",
       "      <td>female</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>STON/O2. 3101282</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1</td>\n",
       "      <td>female</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>113803</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>C123</td>\n",
       "      <td>S</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>0.0</td>\n",
       "      <td>3</td>\n",
       "      <td>male</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>373450</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   PassengerId  Survived  Pclass     Sex   Age  SibSp  Parch  \\\n",
       "0            1       0.0       3    male  22.0      1      0   \n",
       "1            2       1.0       1  female  38.0      1      0   \n",
       "2            3       1.0       3  female  26.0      0      0   \n",
       "3            4       1.0       1  female  35.0      1      0   \n",
       "4            5       0.0       3    male  35.0      0      0   \n",
       "\n",
       "             Ticket     Fare Cabin Embarked  Title  \n",
       "0         A/5 21171   7.2500   NaN        S      2  \n",
       "1          PC 17599  71.2833   C85        C      1  \n",
       "2  STON/O2. 3101282   7.9250   NaN        S      1  \n",
       "3            113803  53.1000  C123        S      1  \n",
       "4            373450   8.0500   NaN        S      2  "
      ]
     },
     "execution_count": 53,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2020-09-08T17:54:45.688864Z",
     "iopub.status.busy": "2020-09-08T17:54:45.687772Z",
     "iopub.status.idle": "2020-09-08T17:54:45.710545Z",
     "shell.execute_reply": "2020-09-08T17:54:45.709836Z"
    },
    "papermill": {
     "duration": 0.081169,
     "end_time": "2020-09-08T17:54:45.710685",
     "exception": false,
     "start_time": "2020-09-08T17:54:45.629516",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>Embarked</th>\n",
       "      <th>Title_0</th>\n",
       "      <th>Title_1</th>\n",
       "      <th>Title_2</th>\n",
       "      <th>Title_3</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>0.0</td>\n",
       "      <td>3</td>\n",
       "      <td>male</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>A/5 21171</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1</td>\n",
       "      <td>female</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>PC 17599</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>C85</td>\n",
       "      <td>C</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>1.0</td>\n",
       "      <td>3</td>\n",
       "      <td>female</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>STON/O2. 3101282</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1</td>\n",
       "      <td>female</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>113803</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>C123</td>\n",
       "      <td>S</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>0.0</td>\n",
       "      <td>3</td>\n",
       "      <td>male</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>373450</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   PassengerId  Survived  Pclass     Sex   Age  SibSp  Parch  \\\n",
       "0            1       0.0       3    male  22.0      1      0   \n",
       "1            2       1.0       1  female  38.0      1      0   \n",
       "2            3       1.0       3  female  26.0      0      0   \n",
       "3            4       1.0       1  female  35.0      1      0   \n",
       "4            5       0.0       3    male  35.0      0      0   \n",
       "\n",
       "             Ticket     Fare Cabin Embarked  Title_0  Title_1  Title_2  \\\n",
       "0         A/5 21171   7.2500   NaN        S        0        0        1   \n",
       "1          PC 17599  71.2833   C85        C        0        1        0   \n",
       "2  STON/O2. 3101282   7.9250   NaN        S        0        1        0   \n",
       "3            113803  53.1000  C123        S        0        1        0   \n",
       "4            373450   8.0500   NaN        S        0        0        1   \n",
       "\n",
       "   Title_3  \n",
       "0        0  \n",
       "1        0  \n",
       "2        0  \n",
       "3        0  \n",
       "4        0  "
      ]
     },
     "execution_count": 54,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_df = pd.get_dummies(train_df,columns=[\"Title\"])\n",
    "train_df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "execution_count": null,
   "metadata": {
    "papermill": {
     "duration": 0.050305,
     "end_time": "2020-09-08T17:54:45.812259",
     "exception": false,
     "start_time": "2020-09-08T17:54:45.761954",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "#bub kodla title featureını 4 featura bölüyoruz"
   ]
  },
  {
   "cell_type": "markdown",
   "execution_count": null,
   "metadata": {
    "papermill": {
     "duration": 0.050335,
     "end_time": "2020-09-08T17:54:45.913190",
     "exception": false,
     "start_time": "2020-09-08T17:54:45.862855",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "> Family Size\n",
    "parch ve sibsp featuralarını alarak yeni bir feature elde edeceğiz."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2020-09-08T17:54:46.035342Z",
     "iopub.status.busy": "2020-09-08T17:54:46.034231Z",
     "iopub.status.idle": "2020-09-08T17:54:46.038858Z",
     "shell.execute_reply": "2020-09-08T17:54:46.038278Z"
    },
    "papermill": {
     "duration": 0.075401,
     "end_time": "2020-09-08T17:54:46.038984",
     "exception": false,
     "start_time": "2020-09-08T17:54:45.963583",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>Embarked</th>\n",
       "      <th>Title_0</th>\n",
       "      <th>Title_1</th>\n",
       "      <th>Title_2</th>\n",
       "      <th>Title_3</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>0.0</td>\n",
       "      <td>3</td>\n",
       "      <td>male</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>A/5 21171</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1</td>\n",
       "      <td>female</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>PC 17599</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>C85</td>\n",
       "      <td>C</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>1.0</td>\n",
       "      <td>3</td>\n",
       "      <td>female</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>STON/O2. 3101282</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1</td>\n",
       "      <td>female</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>113803</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>C123</td>\n",
       "      <td>S</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>0.0</td>\n",
       "      <td>3</td>\n",
       "      <td>male</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>373450</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   PassengerId  Survived  Pclass     Sex   Age  SibSp  Parch  \\\n",
       "0            1       0.0       3    male  22.0      1      0   \n",
       "1            2       1.0       1  female  38.0      1      0   \n",
       "2            3       1.0       3  female  26.0      0      0   \n",
       "3            4       1.0       1  female  35.0      1      0   \n",
       "4            5       0.0       3    male  35.0      0      0   \n",
       "\n",
       "             Ticket     Fare Cabin Embarked  Title_0  Title_1  Title_2  \\\n",
       "0         A/5 21171   7.2500   NaN        S        0        0        1   \n",
       "1          PC 17599  71.2833   C85        C        0        1        0   \n",
       "2  STON/O2. 3101282   7.9250   NaN        S        0        1        0   \n",
       "3            113803  53.1000  C123        S        0        1        0   \n",
       "4            373450   8.0500   NaN        S        0        0        1   \n",
       "\n",
       "   Title_3  \n",
       "0        0  \n",
       "1        0  \n",
       "2        0  \n",
       "3        0  \n",
       "4        0  "
      ]
     },
     "execution_count": 55,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2020-09-08T17:54:46.148819Z",
     "iopub.status.busy": "2020-09-08T17:54:46.148048Z",
     "iopub.status.idle": "2020-09-08T17:54:46.153263Z",
     "shell.execute_reply": "2020-09-08T17:54:46.152494Z"
    },
    "papermill": {
     "duration": 0.062932,
     "end_time": "2020-09-08T17:54:46.153393",
     "exception": false,
     "start_time": "2020-09-08T17:54:46.090461",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "train_df[\"Fsize\"] = train_df[\"SibSp\"] + train_df[\"Parch\"] + 1              \n",
    "#neden 1 ekledik?bu kişlinin ailesi 0 demek saçma yolcunun kendisi 1 deriz"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2020-09-08T17:54:46.277581Z",
     "iopub.status.busy": "2020-09-08T17:54:46.262700Z",
     "iopub.status.idle": "2020-09-08T17:54:46.282355Z",
     "shell.execute_reply": "2020-09-08T17:54:46.281617Z"
    },
    "papermill": {
     "duration": 0.077813,
     "end_time": "2020-09-08T17:54:46.282481",
     "exception": false,
     "start_time": "2020-09-08T17:54:46.204668",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>Embarked</th>\n",
       "      <th>Title_0</th>\n",
       "      <th>Title_1</th>\n",
       "      <th>Title_2</th>\n",
       "      <th>Title_3</th>\n",
       "      <th>Fsize</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>0.0</td>\n",
       "      <td>3</td>\n",
       "      <td>male</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>A/5 21171</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1</td>\n",
       "      <td>female</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>PC 17599</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>C85</td>\n",
       "      <td>C</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>1.0</td>\n",
       "      <td>3</td>\n",
       "      <td>female</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>STON/O2. 3101282</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1</td>\n",
       "      <td>female</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>113803</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>C123</td>\n",
       "      <td>S</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>0.0</td>\n",
       "      <td>3</td>\n",
       "      <td>male</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>373450</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   PassengerId  Survived  Pclass     Sex   Age  SibSp  Parch  \\\n",
       "0            1       0.0       3    male  22.0      1      0   \n",
       "1            2       1.0       1  female  38.0      1      0   \n",
       "2            3       1.0       3  female  26.0      0      0   \n",
       "3            4       1.0       1  female  35.0      1      0   \n",
       "4            5       0.0       3    male  35.0      0      0   \n",
       "\n",
       "             Ticket     Fare Cabin Embarked  Title_0  Title_1  Title_2  \\\n",
       "0         A/5 21171   7.2500   NaN        S        0        0        1   \n",
       "1          PC 17599  71.2833   C85        C        0        1        0   \n",
       "2  STON/O2. 3101282   7.9250   NaN        S        0        1        0   \n",
       "3            113803  53.1000  C123        S        0        1        0   \n",
       "4            373450   8.0500   NaN        S        0        0        1   \n",
       "\n",
       "   Title_3  Fsize  \n",
       "0        0      2  \n",
       "1        0      2  \n",
       "2        0      1  \n",
       "3        0      2  \n",
       "4        0      1  "
      ]
     },
     "execution_count": 57,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2020-09-08T17:54:46.394802Z",
     "iopub.status.busy": "2020-09-08T17:54:46.393031Z",
     "iopub.status.idle": "2020-09-08T17:54:46.855200Z",
     "shell.execute_reply": "2020-09-08T17:54:46.854402Z"
    },
    "papermill": {
     "duration": 0.52043,
     "end_time": "2020-09-08T17:54:46.855320",
     "exception": false,
     "start_time": "2020-09-08T17:54:46.334890",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "image/png": "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/zoiIkKdOnVSbm6uY0MBQDhoM8AlJSV6/fXXtXr1al28eFFPPvmkjh8/rubm5kDNBwAhq80A//rXv9bixYslSVu3bpXX69WWLVt05swZTZ06VRkZGQEZEgBCUZvvhLvhhhvUp08fSdJf//pXPfzww3K5XLr55psVFfX1Ry8KCgo0duxYZWdn6+OPP271Nq+++qoef/zxf2N0ALi+tRngxsZGNTc3q6GhQTt27NCwYcP813m93jYfuLS0VFVVVSosLNT8+fM1b968K25z4MABffDBB//m6ABwfWszwN///vf16KOPavTo0Ro2bJi+/e1vq7GxUc8//7yGDBnS5gOXlJQoMzNTktSvXz/V1dWpvr6+xW0WLlyon/3sZ//hUwCA61ObxxHGjRun4cOH6+zZs/5PwYiOjtaQIUM0evToNh/Y4/EoNTXVvx4XF6fa2lrFxMRIkoqKinT33Xfr1ltvbfNxKioq2vVEAqVzALcVTM89WGYJljlCUWNjo3/J9/naSklJafXyrz2Q21ogx4wZ87Ub9Pl8V6xf/mDP06dPq6ioSL/73e9UXV3d5uNcbXAr/wrgttp87u8Hbg7p6rPUBHaMoPvzEEqio6P9S77PgdHuT8ToqMTERHk8Hv96TU2N4uPjJUlut1snT57UuHHjNG3aNH3yyScqKChwahQACEqOBTgtLU3FxcWSpP379yshIcF/+OGBBx7Qe++9p3Xr1um1115Tamoqb+wAEHba9U64f8fgwYOVmpqq7OxsuVwu5efnq6ioSLGxscrKynJqswBw3XAswJI0a9asFuutfZx9r169tHr1aifHAICg5NghCABA2wgwABghwABghAADgBECDABGCDAAGCHAAGCEAAOAEQIMAEYIMAAYIcAAYIQAA4ARAgwARkI6wG63Wzk5OXK73dajAMAVHD0dpbVVq1apsrJSXq9XQ4cOtR4HAFoI6T1gr9fbYgkAwSSkAwwAwSykD0EA4axiwf/p0O0bTzb4lx29b8qcER26PS5hDxgAjBBgADBCgAHACAEGACMEGACMEGAAMEKAAcAIAQYAIwQYAIxcN++E+87sdzp8n1jPWUVK+pfnbIfuX/bfT3R4WwDQUewBA4ARAgwARggwABghwABghAADgBECDABGCDAAGCHAAGCEAAOAEQIMAEYIMAAYIcAAYCSkA+yLiGqxBIBgEtIB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\n",
      "text/plain": [
       "<Figure size 360x360 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "g = sns.factorplot(x = \"Fsize\", y = \"Survived\", data = train_df, kind = \"bar\")\n",
    "g.set_ylabels(\"Survival\")\n",
    "plt.show()\n",
    "#yeni yaptığımız featurun adı fsize"
   ]
  },
  {
   "cell_type": "markdown",
   "execution_count": null,
   "metadata": {
    "papermill": {
     "duration": 0.05133,
     "end_time": "2020-09-08T17:54:46.958722",
     "exception": false,
     "start_time": "2020-09-08T17:54:46.907392",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "familesizemiz 4 e kadar hayatta kalma oranımız yükselirken sonra düşüyor.biz burda ili kategori yapıcaz.iki gruba ayırıcaz.5tan küçük ve büyükler diye"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2020-09-08T17:54:47.071160Z",
     "iopub.status.busy": "2020-09-08T17:54:47.070263Z",
     "iopub.status.idle": "2020-09-08T17:54:47.073383Z",
     "shell.execute_reply": "2020-09-08T17:54:47.073928Z"
    },
    "papermill": {
     "duration": 0.062831,
     "end_time": "2020-09-08T17:54:47.074084",
     "exception": false,
     "start_time": "2020-09-08T17:54:47.011253",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "train_df[\"family_size\"] = [1 if i < 5 else 0 for i in train_df[\"Fsize\"]]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2020-09-08T17:54:47.202539Z",
     "iopub.status.busy": "2020-09-08T17:54:47.201523Z",
     "iopub.status.idle": "2020-09-08T17:54:47.206718Z",
     "shell.execute_reply": "2020-09-08T17:54:47.206110Z"
    },
    "papermill": {
     "duration": 0.081245,
     "end_time": "2020-09-08T17:54:47.206855",
     "exception": false,
     "start_time": "2020-09-08T17:54:47.125610",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>Embarked</th>\n",
       "      <th>Title_0</th>\n",
       "      <th>Title_1</th>\n",
       "      <th>Title_2</th>\n",
       "      <th>Title_3</th>\n",
       "      <th>Fsize</th>\n",
       "      <th>family_size</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>0.0</td>\n",
       "      <td>3</td>\n",
       "      <td>male</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>A/5 21171</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1</td>\n",
       "      <td>female</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>PC 17599</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>C85</td>\n",
       "      <td>C</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>1.0</td>\n",
       "      <td>3</td>\n",
       "      <td>female</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>STON/O2. 3101282</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1</td>\n",
       "      <td>female</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>113803</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>C123</td>\n",
       "      <td>S</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>0.0</td>\n",
       "      <td>3</td>\n",
       "      <td>male</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>373450</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>6</td>\n",
       "      <td>0.0</td>\n",
       "      <td>3</td>\n",
       "      <td>male</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>330877</td>\n",
       "      <td>8.4583</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Q</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>7</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1</td>\n",
       "      <td>male</td>\n",
       "      <td>54.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>17463</td>\n",
       "      <td>51.8625</td>\n",
       "      <td>E46</td>\n",
       "      <td>S</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>8</td>\n",
       "      <td>0.0</td>\n",
       "      <td>3</td>\n",
       "      <td>male</td>\n",
       "      <td>2.0</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>349909</td>\n",
       "      <td>21.0750</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>5</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>9</td>\n",
       "      <td>1.0</td>\n",
       "      <td>3</td>\n",
       "      <td>female</td>\n",
       "      <td>27.0</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>347742</td>\n",
       "      <td>11.1333</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>10</td>\n",
       "      <td>1.0</td>\n",
       "      <td>2</td>\n",
       "      <td>female</td>\n",
       "      <td>14.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>237736</td>\n",
       "      <td>30.0708</td>\n",
       "      <td>NaN</td>\n",
       "      <td>C</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   PassengerId  Survived  Pclass     Sex   Age  SibSp  Parch  \\\n",
       "0            1       0.0       3    male  22.0      1      0   \n",
       "1            2       1.0       1  female  38.0      1      0   \n",
       "2            3       1.0       3  female  26.0      0      0   \n",
       "3            4       1.0       1  female  35.0      1      0   \n",
       "4            5       0.0       3    male  35.0      0      0   \n",
       "5            6       0.0       3    male   NaN      0      0   \n",
       "6            7       0.0       1    male  54.0      0      0   \n",
       "7            8       0.0       3    male   2.0      3      1   \n",
       "8            9       1.0       3  female  27.0      0      2   \n",
       "9           10       1.0       2  female  14.0      1      0   \n",
       "\n",
       "             Ticket     Fare Cabin Embarked  Title_0  Title_1  Title_2  \\\n",
       "0         A/5 21171   7.2500   NaN        S        0        0        1   \n",
       "1          PC 17599  71.2833   C85        C        0        1        0   \n",
       "2  STON/O2. 3101282   7.9250   NaN        S        0        1        0   \n",
       "3            113803  53.1000  C123        S        0        1        0   \n",
       "4            373450   8.0500   NaN        S        0        0        1   \n",
       "5            330877   8.4583   NaN        Q        0        0        1   \n",
       "6             17463  51.8625   E46        S        0        0        1   \n",
       "7            349909  21.0750   NaN        S        1        0        0   \n",
       "8            347742  11.1333   NaN        S        0        1        0   \n",
       "9            237736  30.0708   NaN        C        0        1        0   \n",
       "\n",
       "   Title_3  Fsize  family_size  \n",
       "0        0      2            1  \n",
       "1        0      2            1  \n",
       "2        0      1            1  \n",
       "3        0      2            1  \n",
       "4        0      1            1  \n",
       "5        0      1            1  \n",
       "6        0      1            1  \n",
       "7        0      5            0  \n",
       "8        0      3            1  \n",
       "9        0      2            1  "
      ]
     },
     "execution_count": 60,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_df.head(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2020-09-08T17:54:47.319283Z",
     "iopub.status.busy": "2020-09-08T17:54:47.318171Z",
     "iopub.status.idle": "2020-09-08T17:54:47.435168Z",
     "shell.execute_reply": "2020-09-08T17:54:47.434419Z"
    },
    "papermill": {
     "duration": 0.175897,
     "end_time": "2020-09-08T17:54:47.435286",
     "exception": false,
     "start_time": "2020-09-08T17:54:47.259389",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.countplot(x = \"family_size\", data = train_df)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2020-09-08T17:54:47.552945Z",
     "iopub.status.busy": "2020-09-08T17:54:47.546837Z",
     "iopub.status.idle": "2020-09-08T17:54:47.785285Z",
     "shell.execute_reply": "2020-09-08T17:54:47.784651Z"
    },
    "papermill": {
     "duration": 0.297616,
     "end_time": "2020-09-08T17:54:47.785409",
     "exception": false,
     "start_time": "2020-09-08T17:54:47.487793",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAWAAAAFgCAYAAACFYaNMAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjEsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+j8jraAAAU1ElEQVR4nO3df2xV9f3H8deF266TFpT+UmRkbM1sdzcNLSGaUlm0RSbiVKb3Mn4Icxi0DibjOygbNNquk8F+WFYT8Y8ximLRNUuWObvAhInerqwQkJRkdLIWzLD3KqVcymwr9/vHwo21pVxbTt+X2+fjn9tzz+257za3z5x8en+4wuFwWACAYTfKegAAGKkIMAAYIcAAYIQAA4ARAgwARmI6wI2NjdYjAIBjYjrAABDPCDAAGCHAAGCEAAOAEQIMAEYIMAAYIcAAYIQAA4ARAgwARggwABghwABghAADgBECDABGCDCAqNXX12vlypWqr6+3HiUuuK0HAHD12Lp1q44dO6bOzk7deuut1uNc9TgDBhC1zs7OXpcYGgIMAEYIMAAYIcAAYIQAA4ARAgwARggwABghwABghAADgBECDABGCDAAGCHAAGCEAAOAEQIMAEYIMAAYIcAAYIQAA4ARAgwARggwABhxNMAVFRXyer3y+Xw6fPhwv7f5xS9+oYULFzo5BgDEJMc+lLOhoUEtLS2qqalRc3OzSkpK9Morr/S6TXNzs/bv36+EhASnxgCAmOXYGbDf71dhYaEkKSsrSx0dHQqFQr1u88wzz+jJJ590agQAiGmOnQEHg0F5PJ7IdmpqqgKBgJKTkyVJtbW1mjZtmm688UanRgAc1fr0161HGHY9H46X5FbPhy0j6ueftP4dR47rWIDD4XCfbZfLJUlqb29XbW2tfvvb3+r9998f8DhHjx51akRgSMZYD4BhM9QO5eTk9Hu9YwHOzMxUMBiMbLe1tSktLU2SVF9frw8//FDz589XV1eXWltbVVFRobVr1/Y5zqUGB6y1Wg+AYeNUhxxbA87Pz1ddXZ0kqampSRkZGZHlh1mzZum1117Tzp079Zvf/EYej6ff+AJAPHPsDDg3N1cej0c+n08ul0ulpaWqra1VSkqKioqKnLpbALhqOBZgSVq1alWv7ezs7D63mThxoqqrq50cAwBiEq+EAwAjBBgAjBBgADBCgAHACAEGACMEGACMEGAAMEKAAcAIAQYAIwQYAIwQYAAwQoABwAgBBgAjBBgAjBBgADBCgAHACAEGACMEGEDUkkaHe11iaAgwgKjd/8Vzyh7Xpfu/eM56lLjg6GfCAYgvt6R26ZbULusx4gZnwABghAADgBECDABGCDAAGCHAAGCEAAOAEQIMAEYIMAAYIcAAYIQAA4ARAgwARggwABghwABghAADgBECDABGCDAAGCHAAGCEAAOAEQIMAEYIMAAYIcAAYIQAA4ARAgwARggwABghwABghAADgBECDABGCDAAGCHAAGCEAAOAEQIMAEYIMAAYIcAAYIQAA4ARAgwARggwABghwABghAADgBECDABGCDAAGHE7efCKigodOnRILpdLa9eu1c033xzZt3PnTr366qsaNWqUsrOzVVpaKpfL5eQ4ABBTHDsDbmhoUEtLi2pqalReXq6ysrLIvvPnz+tPf/qTXnzxRb388st69913dfDgQadGAYCY5FiA/X6/CgsLJUlZWVnq6OhQKBSSJH3+85/X7373OyUkJOj8+fMKhUJKT093ahQAiEmOLUEEg0F5PJ7IdmpqqgKBgJKTkyPXbdmyRdu2bdOiRYv0hS98od/jHD161KkRgSEZYz0Ahs1QO5STk9Pv9Y4FOBwO99n+9Brvo48+qkWLFmnp0qXKy8tTXl5en+NcanDAWqv1ABg2TnXIsSWIzMxMBYPByHZbW5vS0tIkSe3t7dq/f78kKSkpSbfffrsOHDjg1CgAEJMcC3B+fr7q6uokSU1NTcrIyIgsP/T09GjNmjU6d+6cJOmdd97R5MmTnRoFAGKSY0sQubm58ng88vl8crlcKi0tVW1trVJSUlRUVKTi4mItWrRIbrdbN910k+68806nRgGAmOQKf3qxNoY0Njb2uy4MxILWp79uPQKGyaT17zhyXF4JBwBGCDAAGCHAAGCEAAOAEQIMAEYIMAAYIcAAYIQAA4ARAgwARggwABghwABghAADgBECDABGCDAAGCHAAGCEAAOAEQIMAEYIMAAYIcAAYIQAA4ARAgwARggwABghwABghAADgBECDABGCDAAGCHAAGCEAAOAEfdAO5cvXy6Xy3XJ/c8+++wVHwgARooBA7xgwYJL7gsGg1d8GAAYSQYM8LRp0yRJPT092rdvn9rb2yVJ3d3dev7553X3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\n",
      "text/plain": [
       "<Figure size 360x360 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "g = sns.factorplot(x = \"family_size\", y = \"Survived\", data = train_df, kind = \"bar\")\n",
    "g.set_ylabels(\"Survival\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "execution_count": null,
   "metadata": {
    "papermill": {
     "duration": 0.052705,
     "end_time": "2020-09-08T17:54:47.891777",
     "exception": false,
     "start_time": "2020-09-08T17:54:47.839072",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "büyük aileler hayatta kalma olasılığı daha düşük ama daha az kilşili küçük ailelerin hayatta kalma olasılığı daha yüksek."
   ]
  },
  {
   "cell_type": "markdown",
   "execution_count": null,
   "metadata": {
    "papermill": {
     "duration": 0.053769,
     "end_time": "2020-09-08T17:54:47.998917",
     "exception": false,
     "start_time": "2020-09-08T17:54:47.945148",
     "status": "completed"
    },
    "tags": []
   },
   "source": []
  },
  {
   "cell_type": "markdown",
   "execution_count": null,
   "metadata": {
    "papermill": {
     "duration": 0.052845,
     "end_time": "2020-09-08T17:54:48.105706",
     "exception": false,
     "start_time": "2020-09-08T17:54:48.052861",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "Small familes have more chance to survive than large families."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2020-09-08T17:54:48.226128Z",
     "iopub.status.busy": "2020-09-08T17:54:48.225326Z",
     "iopub.status.idle": "2020-09-08T17:54:48.245043Z",
     "shell.execute_reply": "2020-09-08T17:54:48.244314Z"
    },
    "papermill": {
     "duration": 0.086166,
     "end_time": "2020-09-08T17:54:48.245166",
     "exception": false,
     "start_time": "2020-09-08T17:54:48.159000",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>Embarked</th>\n",
       "      <th>Title_0</th>\n",
       "      <th>Title_1</th>\n",
       "      <th>Title_2</th>\n",
       "      <th>Title_3</th>\n",
       "      <th>Fsize</th>\n",
       "      <th>family_size_0</th>\n",
       "      <th>family_size_1</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>0.0</td>\n",
       "      <td>3</td>\n",
       "      <td>male</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>A/5 21171</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1</td>\n",
       "      <td>female</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>PC 17599</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>C85</td>\n",
       "      <td>C</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>1.0</td>\n",
       "      <td>3</td>\n",
       "      <td>female</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>STON/O2. 3101282</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1</td>\n",
       "      <td>female</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>113803</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>C123</td>\n",
       "      <td>S</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>0.0</td>\n",
       "      <td>3</td>\n",
       "      <td>male</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>373450</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   PassengerId  Survived  Pclass     Sex   Age  SibSp  Parch  \\\n",
       "0            1       0.0       3    male  22.0      1      0   \n",
       "1            2       1.0       1  female  38.0      1      0   \n",
       "2            3       1.0       3  female  26.0      0      0   \n",
       "3            4       1.0       1  female  35.0      1      0   \n",
       "4            5       0.0       3    male  35.0      0      0   \n",
       "\n",
       "             Ticket     Fare Cabin Embarked  Title_0  Title_1  Title_2  \\\n",
       "0         A/5 21171   7.2500   NaN        S        0        0        1   \n",
       "1          PC 17599  71.2833   C85        C        0        1        0   \n",
       "2  STON/O2. 3101282   7.9250   NaN        S        0        1        0   \n",
       "3            113803  53.1000  C123        S        0        1        0   \n",
       "4            373450   8.0500   NaN        S        0        0        1   \n",
       "\n",
       "   Title_3  Fsize  family_size_0  family_size_1  \n",
       "0        0      2              0              1  \n",
       "1        0      2              0              1  \n",
       "2        0      1              0              1  \n",
       "3        0      2              0              1  \n",
       "4        0      1              0              1  "
      ]
     },
     "execution_count": 63,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_df = pd.get_dummies(train_df, columns= [\"family_size\"])\n",
    "train_df.head()\n",
    "#familysizeı iki kategoriye bölmek için"
   ]
  },
  {
   "cell_type": "markdown",
   "execution_count": null,
   "metadata": {
    "papermill": {
     "duration": 0.053325,
     "end_time": "2020-09-08T17:54:48.351966",
     "exception": false,
     "start_time": "2020-09-08T17:54:48.298641",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "Embarked yolcularımızın titaniğe nereden bindiği\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2020-09-08T17:54:48.466211Z",
     "iopub.status.busy": "2020-09-08T17:54:48.465336Z",
     "iopub.status.idle": "2020-09-08T17:54:48.470106Z",
     "shell.execute_reply": "2020-09-08T17:54:48.469384Z"
    },
    "papermill": {
     "duration": 0.064812,
     "end_time": "2020-09-08T17:54:48.470226",
     "exception": false,
     "start_time": "2020-09-08T17:54:48.405414",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    S\n",
       "1    C\n",
       "2    S\n",
       "3    S\n",
       "4    S\n",
       "Name: Embarked, dtype: object"
      ]
     },
     "execution_count": 64,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_df[\"Embarked\"].head()"
   ]
  },
  {
   "cell_type": "markdown",
   "execution_count": null,
   "metadata": {
    "papermill": {
     "duration": 0.054257,
     "end_time": "2020-09-08T17:54:48.578137",
     "exception": false,
     "start_time": "2020-09-08T17:54:48.523880",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "3 tane limanımız var olduğu gibi kullanacağız"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2020-09-08T17:54:48.700296Z",
     "iopub.status.busy": "2020-09-08T17:54:48.695534Z",
     "iopub.status.idle": "2020-09-08T17:54:48.824230Z",
     "shell.execute_reply": "2020-09-08T17:54:48.824800Z"
    },
    "papermill": {
     "duration": 0.193165,
     "end_time": "2020-09-08T17:54:48.824974",
     "exception": false,
     "start_time": "2020-09-08T17:54:48.631809",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.countplot(x = \"Embarked\", data = train_df)\n",
    "plt.show()\n",
    "#kaç tane var ona bakcaz"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2020-09-08T17:54:48.960494Z",
     "iopub.status.busy": "2020-09-08T17:54:48.959656Z",
     "iopub.status.idle": "2020-09-08T17:54:48.986550Z",
     "shell.execute_reply": "2020-09-08T17:54:48.987157Z"
    },
    "papermill": {
     "duration": 0.096692,
     "end_time": "2020-09-08T17:54:48.987308",
     "exception": false,
     "start_time": "2020-09-08T17:54:48.890616",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>Title_0</th>\n",
       "      <th>Title_1</th>\n",
       "      <th>Title_2</th>\n",
       "      <th>Title_3</th>\n",
       "      <th>Fsize</th>\n",
       "      <th>family_size_0</th>\n",
       "      <th>family_size_1</th>\n",
       "      <th>Embarked_C</th>\n",
       "      <th>Embarked_Q</th>\n",
       "      <th>Embarked_S</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>0.0</td>\n",
       "      <td>3</td>\n",
       "      <td>male</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>A/5 21171</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1</td>\n",
       "      <td>female</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>PC 17599</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>C85</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>1.0</td>\n",
       "      <td>3</td>\n",
       "      <td>female</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>STON/O2. 3101282</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1</td>\n",
       "      <td>female</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>113803</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>C123</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>0.0</td>\n",
       "      <td>3</td>\n",
       "      <td>male</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>373450</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   PassengerId  Survived  Pclass     Sex   Age  SibSp  Parch  \\\n",
       "0            1       0.0       3    male  22.0      1      0   \n",
       "1            2       1.0       1  female  38.0      1      0   \n",
       "2            3       1.0       3  female  26.0      0      0   \n",
       "3            4       1.0       1  female  35.0      1      0   \n",
       "4            5       0.0       3    male  35.0      0      0   \n",
       "\n",
       "             Ticket     Fare Cabin  Title_0  Title_1  Title_2  Title_3  Fsize  \\\n",
       "0         A/5 21171   7.2500   NaN        0        0        1        0      2   \n",
       "1          PC 17599  71.2833   C85        0        1        0        0      2   \n",
       "2  STON/O2. 3101282   7.9250   NaN        0        1        0        0      1   \n",
       "3            113803  53.1000  C123        0        1        0        0      2   \n",
       "4            373450   8.0500   NaN        0        0        1        0      1   \n",
       "\n",
       "   family_size_0  family_size_1  Embarked_C  Embarked_Q  Embarked_S  \n",
       "0              0              1           0           0           1  \n",
       "1              0              1           1           0           0  \n",
       "2              0              1           0           0           1  \n",
       "3              0              1           0           0           1  \n",
       "4              0              1           0           0           1  "
      ]
     },
     "execution_count": 66,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_df = pd.get_dummies(train_df, columns=[\"Embarked\"])\n",
    "train_df.head()\n",
    "#embarked featurunı kullanabilri hale geticez yani embarkedi ortadana kaldırıp 3 tane(c,q,s)yeni featur ortaya çıkaracağız"
   ]
  },
  {
   "cell_type": "markdown",
   "execution_count": null,
   "metadata": {
    "papermill": {
     "duration": 0.053791,
     "end_time": "2020-09-08T17:54:49.095955",
     "exception": false,
     "start_time": "2020-09-08T17:54:49.042164",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "Ticket"
   ]
  },
  {
   "cell_type": "markdown",
   "execution_count": null,
   "metadata": {
    "papermill": {
     "duration": 0.053997,
     "end_time": "2020-09-08T17:54:49.204293",
     "exception": false,
     "start_time": "2020-09-08T17:54:49.150296",
     "status": "completed"
    },
    "tags": []
   },
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2020-09-08T17:54:49.320536Z",
     "iopub.status.busy": "2020-09-08T17:54:49.317852Z",
     "iopub.status.idle": "2020-09-08T17:54:49.324424Z",
     "shell.execute_reply": "2020-09-08T17:54:49.325030Z"
    },
    "papermill": {
     "duration": 0.066648,
     "end_time": "2020-09-08T17:54:49.325184",
     "exception": false,
     "start_time": "2020-09-08T17:54:49.258536",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0            A/5 21171\n",
       "1             PC 17599\n",
       "2     STON/O2. 3101282\n",
       "3               113803\n",
       "4               373450\n",
       "5               330877\n",
       "6                17463\n",
       "7               349909\n",
       "8               347742\n",
       "9               237736\n",
       "10             PP 9549\n",
       "11              113783\n",
       "12           A/5. 2151\n",
       "13              347082\n",
       "14              350406\n",
       "15              248706\n",
       "16              382652\n",
       "17              244373\n",
       "18              345763\n",
       "19                2649\n",
       "Name: Ticket, dtype: object"
      ]
     },
     "execution_count": 67,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_df[\"Ticket\"].head(20)\n",
    "#burda biletlerin üzerinde harf ve sayılar vae başlarında birbirlerini tekrarlayan sayılar var\n",
    "#bzizde baştakilerle sonu ayıracağız"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2020-09-08T17:54:49.441154Z",
     "iopub.status.busy": "2020-09-08T17:54:49.440270Z",
     "iopub.status.idle": "2020-09-08T17:54:49.444694Z",
     "shell.execute_reply": "2020-09-08T17:54:49.444044Z"
    },
    "papermill": {
     "duration": 0.064949,
     "end_time": "2020-09-08T17:54:49.444812",
     "exception": false,
     "start_time": "2020-09-08T17:54:49.379863",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'A5'"
      ]
     },
     "execution_count": 68,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "a = \"A/5. 2151\"\n",
    "a.replace(\".\",\"\").replace(\"/\",\"\").strip().split(\" \")[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2020-09-08T17:54:49.564181Z",
     "iopub.status.busy": "2020-09-08T17:54:49.563126Z",
     "iopub.status.idle": "2020-09-08T17:54:49.565831Z",
     "shell.execute_reply": "2020-09-08T17:54:49.566397Z"
    },
    "papermill": {
     "duration": 0.067234,
     "end_time": "2020-09-08T17:54:49.566545",
     "exception": false,
     "start_time": "2020-09-08T17:54:49.499311",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "tickets = []\n",
    "for i in list(train_df.Ticket):\n",
    "    if not i.isdigit():\n",
    "        tickets.append(i.replace(\".\",\"\").replace(\"/\",\"\").strip().split(\" \")[0])\n",
    "    else:\n",
    "        tickets.append(\"x\")\n",
    "train_df[\"Ticket\"] = tickets"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 70,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2020-09-08T17:54:49.684380Z",
     "iopub.status.busy": "2020-09-08T17:54:49.683533Z",
     "iopub.status.idle": "2020-09-08T17:54:49.687263Z",
     "shell.execute_reply": "2020-09-08T17:54:49.687809Z"
    },
    "papermill": {
     "duration": 0.066186,
     "end_time": "2020-09-08T17:54:49.687965",
     "exception": false,
     "start_time": "2020-09-08T17:54:49.621779",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0         A5\n",
       "1         PC\n",
       "2     STONO2\n",
       "3          x\n",
       "4          x\n",
       "5          x\n",
       "6          x\n",
       "7          x\n",
       "8          x\n",
       "9          x\n",
       "10        PP\n",
       "11         x\n",
       "12        A5\n",
       "13         x\n",
       "14         x\n",
       "15         x\n",
       "16         x\n",
       "17         x\n",
       "18         x\n",
       "19         x\n",
       "Name: Ticket, dtype: object"
      ]
     },
     "execution_count": 70,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_df[\"Ticket\"].head(20)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2020-09-08T17:54:49.812048Z",
     "iopub.status.busy": "2020-09-08T17:54:49.805191Z",
     "iopub.status.idle": "2020-09-08T17:54:49.829413Z",
     "shell.execute_reply": "2020-09-08T17:54:49.828846Z"
    },
    "papermill": {
     "duration": 0.085807,
     "end_time": "2020-09-08T17:54:49.829534",
     "exception": false,
     "start_time": "2020-09-08T17:54:49.743727",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>Title_0</th>\n",
       "      <th>Title_1</th>\n",
       "      <th>Title_2</th>\n",
       "      <th>Title_3</th>\n",
       "      <th>Fsize</th>\n",
       "      <th>family_size_0</th>\n",
       "      <th>family_size_1</th>\n",
       "      <th>Embarked_C</th>\n",
       "      <th>Embarked_Q</th>\n",
       "      <th>Embarked_S</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>0.0</td>\n",
       "      <td>3</td>\n",
       "      <td>male</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>A5</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1</td>\n",
       "      <td>female</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>PC</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>C85</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>1.0</td>\n",
       "      <td>3</td>\n",
       "      <td>female</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>STONO2</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1</td>\n",
       "      <td>female</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>x</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>C123</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>0.0</td>\n",
       "      <td>3</td>\n",
       "      <td>male</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>x</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   PassengerId  Survived  Pclass     Sex   Age  SibSp  Parch  Ticket     Fare  \\\n",
       "0            1       0.0       3    male  22.0      1      0      A5   7.2500   \n",
       "1            2       1.0       1  female  38.0      1      0      PC  71.2833   \n",
       "2            3       1.0       3  female  26.0      0      0  STONO2   7.9250   \n",
       "3            4       1.0       1  female  35.0      1      0       x  53.1000   \n",
       "4            5       0.0       3    male  35.0      0      0       x   8.0500   \n",
       "\n",
       "  Cabin  Title_0  Title_1  Title_2  Title_3  Fsize  family_size_0  \\\n",
       "0   NaN        0        0        1        0      2              0   \n",
       "1   C85        0        1        0        0      2              0   \n",
       "2   NaN        0        1        0        0      1              0   \n",
       "3  C123        0        1        0        0      2              0   \n",
       "4   NaN        0        0        1        0      1              0   \n",
       "\n",
       "   family_size_1  Embarked_C  Embarked_Q  Embarked_S  \n",
       "0              1           0           0           1  \n",
       "1              1           1           0           0  \n",
       "2              1           0           0           1  \n",
       "3              1           0           0           1  \n",
       "4              1           0           0           1  "
      ]
     },
     "execution_count": 71,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2020-09-08T17:54:49.949464Z",
     "iopub.status.busy": "2020-09-08T17:54:49.948374Z",
     "iopub.status.idle": "2020-09-08T17:54:49.987180Z",
     "shell.execute_reply": "2020-09-08T17:54:49.986425Z"
    },
    "papermill": {
     "duration": 0.101744,
     "end_time": "2020-09-08T17:54:49.987303",
     "exception": false,
     "start_time": "2020-09-08T17:54:49.885559",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>Title_0</th>\n",
       "      <th>...</th>\n",
       "      <th>T_SOTONO2</th>\n",
       "      <th>T_SOTONOQ</th>\n",
       "      <th>T_SP</th>\n",
       "      <th>T_STONO</th>\n",
       "      <th>T_STONO2</th>\n",
       "      <th>T_STONOQ</th>\n",
       "      <th>T_SWPP</th>\n",
       "      <th>T_WC</th>\n",
       "      <th>T_WEP</th>\n",
       "      <th>T_x</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>0.0</td>\n",
       "      <td>3</td>\n",
       "      <td>male</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1</td>\n",
       "      <td>female</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>C85</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>1.0</td>\n",
       "      <td>3</td>\n",
       "      <td>female</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1</td>\n",
       "      <td>female</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>C123</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>0.0</td>\n",
       "      <td>3</td>\n",
       "      <td>male</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>6</td>\n",
       "      <td>0.0</td>\n",
       "      <td>3</td>\n",
       "      <td>male</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>8.4583</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>7</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1</td>\n",
       "      <td>male</td>\n",
       "      <td>54.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>51.8625</td>\n",
       "      <td>E46</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>8</td>\n",
       "      <td>0.0</td>\n",
       "      <td>3</td>\n",
       "      <td>male</td>\n",
       "      <td>2.0</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>21.0750</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>9</td>\n",
       "      <td>1.0</td>\n",
       "      <td>3</td>\n",
       "      <td>female</td>\n",
       "      <td>27.0</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>11.1333</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>10</td>\n",
       "      <td>1.0</td>\n",
       "      <td>2</td>\n",
       "      <td>female</td>\n",
       "      <td>14.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>30.0708</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>10 rows × 56 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "   PassengerId  Survived  Pclass     Sex   Age  SibSp  Parch     Fare Cabin  \\\n",
       "0            1       0.0       3    male  22.0      1      0   7.2500   NaN   \n",
       "1            2       1.0       1  female  38.0      1      0  71.2833   C85   \n",
       "2            3       1.0       3  female  26.0      0      0   7.9250   NaN   \n",
       "3            4       1.0       1  female  35.0      1      0  53.1000  C123   \n",
       "4            5       0.0       3    male  35.0      0      0   8.0500   NaN   \n",
       "5            6       0.0       3    male   NaN      0      0   8.4583   NaN   \n",
       "6            7       0.0       1    male  54.0      0      0  51.8625   E46   \n",
       "7            8       0.0       3    male   2.0      3      1  21.0750   NaN   \n",
       "8            9       1.0       3  female  27.0      0      2  11.1333   NaN   \n",
       "9           10       1.0       2  female  14.0      1      0  30.0708   NaN   \n",
       "\n",
       "   Title_0  ...  T_SOTONO2  T_SOTONOQ  T_SP  T_STONO  T_STONO2  T_STONOQ  \\\n",
       "0        0  ...          0          0     0        0         0         0   \n",
       "1        0  ...          0          0     0        0         0         0   \n",
       "2        0  ...          0          0     0        0         1         0   \n",
       "3        0  ...          0          0     0        0         0         0   \n",
       "4        0  ...          0          0     0        0         0         0   \n",
       "5        0  ...          0          0     0        0         0         0   \n",
       "6        0  ...          0          0     0        0         0         0   \n",
       "7        1  ...          0          0     0        0         0         0   \n",
       "8        0  ...          0          0     0        0         0         0   \n",
       "9        0  ...          0          0     0        0         0         0   \n",
       "\n",
       "   T_SWPP  T_WC  T_WEP  T_x  \n",
       "0       0     0      0    0  \n",
       "1       0     0      0    0  \n",
       "2       0     0      0    0  \n",
       "3       0     0      0    1  \n",
       "4       0     0      0    1  \n",
       "5       0     0      0    1  \n",
       "6       0     0      0    1  \n",
       "7       0     0      0    1  \n",
       "8       0     0      0    1  \n",
       "9       0     0      0    1  \n",
       "\n",
       "[10 rows x 56 columns]"
      ]
     },
     "execution_count": 72,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_df = pd.get_dummies(train_df, columns= [\"Ticket\"], prefix = \"T\")                                                                                 #ticketı kulllanma ticket yerine t yi kullan\n",
    "train_df.head(10)                                                                                                                                #ticketları kategori haline getircez"
   ]
  },
  {
   "cell_type": "markdown",
   "execution_count": null,
   "metadata": {
    "papermill": {
     "duration": 0.056215,
     "end_time": "2020-09-08T17:54:50.100390",
     "exception": false,
     "start_time": "2020-09-08T17:54:50.044175",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "Pclass"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2020-09-08T17:54:50.219166Z",
     "iopub.status.busy": "2020-09-08T17:54:50.218082Z",
     "iopub.status.idle": "2020-09-08T17:54:50.346810Z",
     "shell.execute_reply": "2020-09-08T17:54:50.346153Z"
    },
    "papermill": {
     "duration": 0.190274,
     "end_time": "2020-09-08T17:54:50.346945",
     "exception": false,
     "start_time": "2020-09-08T17:54:50.156671",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.countplot(x = \"Pclass\", data = train_df)\n",
    "plt.show()\n",
    "#burda pclassın içindeki dağılımlara bakıyoruz"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 74,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2020-09-08T17:54:50.468456Z",
     "iopub.status.busy": "2020-09-08T17:54:50.467447Z",
     "iopub.status.idle": "2020-09-08T17:54:50.500863Z",
     "shell.execute_reply": "2020-09-08T17:54:50.500236Z"
    },
    "papermill": {
     "duration": 0.097623,
     "end_time": "2020-09-08T17:54:50.501014",
     "exception": false,
     "start_time": "2020-09-08T17:54:50.403391",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>Title_0</th>\n",
       "      <th>Title_1</th>\n",
       "      <th>...</th>\n",
       "      <th>T_STONO</th>\n",
       "      <th>T_STONO2</th>\n",
       "      <th>T_STONOQ</th>\n",
       "      <th>T_SWPP</th>\n",
       "      <th>T_WC</th>\n",
       "      <th>T_WEP</th>\n",
       "      <th>T_x</th>\n",
       "      <th>Pclass_1</th>\n",
       "      <th>Pclass_2</th>\n",
       "      <th>Pclass_3</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>0.0</td>\n",
       "      <td>male</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>1.0</td>\n",
       "      <td>female</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>C85</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>1.0</td>\n",
       "      <td>female</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>1.0</td>\n",
       "      <td>female</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>C123</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>0.0</td>\n",
       "      <td>male</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 58 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "   PassengerId  Survived     Sex   Age  SibSp  Parch     Fare Cabin  Title_0  \\\n",
       "0            1       0.0    male  22.0      1      0   7.2500   NaN        0   \n",
       "1            2       1.0  female  38.0      1      0  71.2833   C85        0   \n",
       "2            3       1.0  female  26.0      0      0   7.9250   NaN        0   \n",
       "3            4       1.0  female  35.0      1      0  53.1000  C123        0   \n",
       "4            5       0.0    male  35.0      0      0   8.0500   NaN        0   \n",
       "\n",
       "   Title_1  ...  T_STONO  T_STONO2  T_STONOQ  T_SWPP  T_WC  T_WEP  T_x  \\\n",
       "0        0  ...        0         0         0       0     0      0    0   \n",
       "1        1  ...        0         0         0       0     0      0    0   \n",
       "2        1  ...        0         1         0       0     0      0    0   \n",
       "3        1  ...        0         0         0       0     0      0    1   \n",
       "4        0  ...        0         0         0       0     0      0    1   \n",
       "\n",
       "   Pclass_1  Pclass_2  Pclass_3  \n",
       "0         0         0         1  \n",
       "1         1         0         0  \n",
       "2         0         0         1  \n",
       "3         1         0         0  \n",
       "4         0         0         1  \n",
       "\n",
       "[5 rows x 58 columns]"
      ]
     },
     "execution_count": 74,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_df[\"Pclass\"] = train_df[\"Pclass\"].astype(\"category\")\n",
    "train_df = pd.get_dummies(train_df, columns= [\"Pclass\"])\n",
    "train_df.head()                                                                                                                               #bu dağılımları pclass1 pclass2 ve pclass3 diye farklı featurelara dağıtalım"
   ]
  },
  {
   "cell_type": "markdown",
   "execution_count": null,
   "metadata": {
    "papermill": {
     "duration": 0.057321,
     "end_time": "2020-09-08T17:54:50.615543",
     "exception": false,
     "start_time": "2020-09-08T17:54:50.558222",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "Sex"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 75,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2020-09-08T17:54:50.738141Z",
     "iopub.status.busy": "2020-09-08T17:54:50.735402Z",
     "iopub.status.idle": "2020-09-08T17:54:50.767784Z",
     "shell.execute_reply": "2020-09-08T17:54:50.768416Z"
    },
    "papermill": {
     "duration": 0.095571,
     "end_time": "2020-09-08T17:54:50.768571",
     "exception": false,
     "start_time": "2020-09-08T17:54:50.673000",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>Title_0</th>\n",
       "      <th>Title_1</th>\n",
       "      <th>Title_2</th>\n",
       "      <th>...</th>\n",
       "      <th>T_STONOQ</th>\n",
       "      <th>T_SWPP</th>\n",
       "      <th>T_WC</th>\n",
       "      <th>T_WEP</th>\n",
       "      <th>T_x</th>\n",
       "      <th>Pclass_1</th>\n",
       "      <th>Pclass_2</th>\n",
       "      <th>Pclass_3</th>\n",
       "      <th>Sex_female</th>\n",
       "      <th>Sex_male</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>0.0</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>1.0</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>C85</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>1.0</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>1.0</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>C123</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>0.0</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 59 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "   PassengerId  Survived   Age  SibSp  Parch     Fare Cabin  Title_0  Title_1  \\\n",
       "0            1       0.0  22.0      1      0   7.2500   NaN        0        0   \n",
       "1            2       1.0  38.0      1      0  71.2833   C85        0        1   \n",
       "2            3       1.0  26.0      0      0   7.9250   NaN        0        1   \n",
       "3            4       1.0  35.0      1      0  53.1000  C123        0        1   \n",
       "4            5       0.0  35.0      0      0   8.0500   NaN        0        0   \n",
       "\n",
       "   Title_2  ...  T_STONOQ  T_SWPP  T_WC  T_WEP  T_x  Pclass_1  Pclass_2  \\\n",
       "0        1  ...         0       0     0      0    0         0         0   \n",
       "1        0  ...         0       0     0      0    0         1         0   \n",
       "2        0  ...         0       0     0      0    0         0         0   \n",
       "3        0  ...         0       0     0      0    1         1         0   \n",
       "4        1  ...         0       0     0      0    1         0         0   \n",
       "\n",
       "   Pclass_3  Sex_female  Sex_male  \n",
       "0         1           0         1  \n",
       "1         0           1         0  \n",
       "2         1           1         0  \n",
       "3         0           1         0  \n",
       "4         1           0         1  \n",
       "\n",
       "[5 rows x 59 columns]"
      ]
     },
     "execution_count": 75,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_df[\"Sex\"] = train_df[\"Sex\"].astype(\"category\")\n",
    "train_df = pd.get_dummies(train_df, columns=[\"Sex\"])\n",
    "train_df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "execution_count": null,
   "metadata": {
    "papermill": {
     "duration": 0.057065,
     "end_time": "2020-09-08T17:54:50.882857",
     "exception": false,
     "start_time": "2020-09-08T17:54:50.825792",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "cinsiyet featurenı kullanacağımız bir yer olmayacak sadece burda kullanabilri hake getireceğiz"
   ]
  },
  {
   "cell_type": "markdown",
   "execution_count": null,
   "metadata": {
    "papermill": {
     "duration": 0.056563,
     "end_time": "2020-09-08T17:54:50.996713",
     "exception": false,
     "start_time": "2020-09-08T17:54:50.940150",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "Drop Passenger ID and Cabin"
   ]
  },
  {
   "cell_type": "markdown",
   "execution_count": null,
   "metadata": {
    "papermill": {
     "duration": 0.056752,
     "end_time": "2020-09-08T17:54:51.111223",
     "exception": false,
     "start_time": "2020-09-08T17:54:51.054471",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "passenger ıd ve cabini bir sütun olarak drop ediyoruz"
   ]
  },
  {
   "cell_type": "markdown",
   "execution_count": null,
   "metadata": {
    "papermill": {
     "duration": 0.056559,
     "end_time": "2020-09-08T17:54:51.224672",
     "exception": false,
     "start_time": "2020-09-08T17:54:51.168113",
     "status": "completed"
    },
    "tags": []
   },
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 76,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2020-09-08T17:54:51.443790Z",
     "iopub.status.busy": "2020-09-08T17:54:51.442982Z",
     "iopub.status.idle": "2020-09-08T17:54:51.447894Z",
     "shell.execute_reply": "2020-09-08T17:54:51.447289Z"
    },
    "papermill": {
     "duration": 0.165281,
     "end_time": "2020-09-08T17:54:51.448024",
     "exception": false,
     "start_time": "2020-09-08T17:54:51.282743",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "train_df.drop(labels = [\"PassengerId\", \"Cabin\"], axis = 1, inplace = True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 77,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2020-09-08T17:54:51.569738Z",
     "iopub.status.busy": "2020-09-08T17:54:51.568628Z",
     "iopub.status.idle": "2020-09-08T17:54:51.572524Z",
     "shell.execute_reply": "2020-09-08T17:54:51.573111Z"
    },
    "papermill": {
     "duration": 0.067559,
     "end_time": "2020-09-08T17:54:51.573274",
     "exception": false,
     "start_time": "2020-09-08T17:54:51.505715",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['Survived', 'Age', 'SibSp', 'Parch', 'Fare', 'Title_0', 'Title_1',\n",
       "       'Title_2', 'Title_3', 'Fsize', 'family_size_0', 'family_size_1',\n",
       "       'Embarked_C', 'Embarked_Q', 'Embarked_S', 'T_A', 'T_A4', 'T_A5',\n",
       "       'T_AQ3', 'T_AQ4', 'T_AS', 'T_C', 'T_CA', 'T_CASOTON', 'T_FC', 'T_FCC',\n",
       "       'T_Fa', 'T_LINE', 'T_LP', 'T_PC', 'T_PP', 'T_PPP', 'T_SC', 'T_SCA3',\n",
       "       'T_SCA4', 'T_SCAH', 'T_SCOW', 'T_SCPARIS', 'T_SCParis', 'T_SOC',\n",
       "       'T_SOP', 'T_SOPP', 'T_SOTONO2', 'T_SOTONOQ', 'T_SP', 'T_STONO',\n",
       "       'T_STONO2', 'T_STONOQ', 'T_SWPP', 'T_WC', 'T_WEP', 'T_x', 'Pclass_1',\n",
       "       'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male'],\n",
       "      dtype='object')"
      ]
     },
     "execution_count": 77,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_df.columns"
   ]
  },
  {
   "cell_type": "markdown",
   "execution_count": null,
   "metadata": {
    "papermill": {
     "duration": 0.095655,
     "end_time": "2020-09-08T17:54:51.726996",
     "exception": false,
     "start_time": "2020-09-08T17:54:51.631341",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "Modeling"
   ]
  },
  {
   "cell_type": "markdown",
   "execution_count": null,
   "metadata": {
    "papermill": {
     "duration": 0.057506,
     "end_time": "2020-09-08T17:54:51.842219",
     "exception": false,
     "start_time": "2020-09-08T17:54:51.784713",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "makine öğrenmesi modelini eğiticeğpimiz yer.ilk önce ml modelimizi eğitmek için gerekli olan kütüphaneleri import edeceğiz."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 78,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2020-09-08T17:54:51.964343Z",
     "iopub.status.busy": "2020-09-08T17:54:51.963523Z",
     "iopub.status.idle": "2020-09-08T17:54:52.378254Z",
     "shell.execute_reply": "2020-09-08T17:54:52.377569Z"
    },
    "papermill": {
     "duration": 0.478733,
     "end_time": "2020-09-08T17:54:52.378383",
     "exception": false,
     "start_time": "2020-09-08T17:54:51.899650",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.svm import SVC\n",
    "from sklearn.ensemble import RandomForestClassifier, VotingClassifier\n",
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "from sklearn.tree import DecisionTreeClassifier\n",
    "from sklearn.metrics import accuracy_score"
   ]
  },
  {
   "cell_type": "markdown",
   "execution_count": null,
   "metadata": {
    "papermill": {
     "duration": 0.057635,
     "end_time": "2020-09-08T17:54:52.494179",
     "exception": false,
     "start_time": "2020-09-08T17:54:52.436544",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "Train - Test Split"
   ]
  },
  {
   "cell_type": "markdown",
   "execution_count": null,
   "metadata": {
    "papermill": {
     "duration": 0.057976,
     "end_time": "2020-09-08T17:54:52.610078",
     "exception": false,
     "start_time": "2020-09-08T17:54:52.552102",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "ilk önce train ve test verisetlerimiz ayırmakla başlıyoruz"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 79,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2020-09-08T17:54:52.733874Z",
     "iopub.status.busy": "2020-09-08T17:54:52.732772Z",
     "iopub.status.idle": "2020-09-08T17:54:52.737460Z",
     "shell.execute_reply": "2020-09-08T17:54:52.736832Z"
    },
    "papermill": {
     "duration": 0.067721,
     "end_time": "2020-09-08T17:54:52.737582",
     "exception": false,
     "start_time": "2020-09-08T17:54:52.669861",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "874"
      ]
     },
     "execution_count": 79,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_df_len"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 80,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2020-09-08T17:54:52.862204Z",
     "iopub.status.busy": "2020-09-08T17:54:52.861445Z",
     "iopub.status.idle": "2020-09-08T17:54:52.864175Z",
     "shell.execute_reply": "2020-09-08T17:54:52.864717Z"
    },
    "papermill": {
     "duration": 0.069093,
     "end_time": "2020-09-08T17:54:52.864915",
     "exception": false,
     "start_time": "2020-09-08T17:54:52.795822",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "test = train_df[train_df_len:]\n",
    "test.drop(labels = [\"Survived\"],axis = 1, inplace = True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 81,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2020-09-08T17:54:53.007009Z",
     "iopub.status.busy": "2020-09-08T17:54:53.006176Z",
     "iopub.status.idle": "2020-09-08T17:54:53.011136Z",
     "shell.execute_reply": "2020-09-08T17:54:53.010540Z"
    },
    "papermill": {
     "duration": 0.088072,
     "end_time": "2020-09-08T17:54:53.011262",
     "exception": false,
     "start_time": "2020-09-08T17:54:52.923190",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Title_0</th>\n",
       "      <th>Title_1</th>\n",
       "      <th>Title_2</th>\n",
       "      <th>Title_3</th>\n",
       "      <th>Fsize</th>\n",
       "      <th>family_size_0</th>\n",
       "      <th>...</th>\n",
       "      <th>T_STONOQ</th>\n",
       "      <th>T_SWPP</th>\n",
       "      <th>T_WC</th>\n",
       "      <th>T_WEP</th>\n",
       "      <th>T_x</th>\n",
       "      <th>Pclass_1</th>\n",
       "      <th>Pclass_2</th>\n",
       "      <th>Pclass_3</th>\n",
       "      <th>Sex_female</th>\n",
       "      <th>Sex_male</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>874</th>\n",
       "      <td>34.5</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7.8292</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>875</th>\n",
       "      <td>47.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>7.0000</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>876</th>\n",
       "      <td>62.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>9.6875</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>877</th>\n",
       "      <td>27.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>8.6625</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>878</th>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>12.2875</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 56 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "      Age  SibSp  Parch     Fare  Title_0  Title_1  Title_2  Title_3  Fsize  \\\n",
       "874  34.5      0      0   7.8292        0        0        1        0      1   \n",
       "875  47.0      1      0   7.0000        0        1        0        0      2   \n",
       "876  62.0      0      0   9.6875        0        0        1        0      1   \n",
       "877  27.0      0      0   8.6625        0        0        1        0      1   \n",
       "878  22.0      1      1  12.2875        0        1        0        0      3   \n",
       "\n",
       "     family_size_0  ...  T_STONOQ  T_SWPP  T_WC  T_WEP  T_x  Pclass_1  \\\n",
       "874              0  ...         0       0     0      0    1         0   \n",
       "875              0  ...         0       0     0      0    1         0   \n",
       "876              0  ...         0       0     0      0    1         0   \n",
       "877              0  ...         0       0     0      0    1         0   \n",
       "878              0  ...         0       0     0      0    1         0   \n",
       "\n",
       "     Pclass_2  Pclass_3  Sex_female  Sex_male  \n",
       "874         0         1           0         1  \n",
       "875         0         1           1         0  \n",
       "876         1         0           0         1  \n",
       "877         0         1           0         1  \n",
       "878         0         1           1         0  \n",
       "\n",
       "[5 rows x 56 columns]"
      ]
     },
     "execution_count": 81,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 82,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2020-09-08T17:54:53.139413Z",
     "iopub.status.busy": "2020-09-08T17:54:53.138296Z",
     "iopub.status.idle": "2020-09-08T17:54:53.148054Z",
     "shell.execute_reply": "2020-09-08T17:54:53.147415Z"
    },
    "papermill": {
     "duration": 0.078082,
     "end_time": "2020-09-08T17:54:53.148177",
     "exception": false,
     "start_time": "2020-09-08T17:54:53.070095",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "X_train 585\n",
      "X_test 289\n",
      "y_train 585\n",
      "y_test 289\n",
      "test 418\n"
     ]
    }
   ],
   "source": [
    "train = train_df[:train_df_len]\n",
    "X_train = train.drop(labels = \"Survived\", axis = 1)\n",
    "y_train = train[\"Survived\"]\n",
    "X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.33, random_state = 42)\n",
    "print(\"X_train\",len(X_train))\n",
    "print(\"X_test\",len(X_test))\n",
    "print(\"y_train\",len(y_train))\n",
    "print(\"y_test\",len(y_test))\n",
    "print(\"test\",len(test))"
   ]
  },
  {
   "cell_type": "markdown",
   "execution_count": null,
   "metadata": {
    "papermill": {
     "duration": 0.057748,
     "end_time": "2020-09-08T17:54:53.264277",
     "exception": false,
     "start_time": "2020-09-08T17:54:53.206529",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "bizim burda kalan trainle mli eğitiyorum kalan testke test  modelim hazır dediğindede görmediği test çıkarıyroum"
   ]
  },
  {
   "cell_type": "markdown",
   "execution_count": null,
   "metadata": {
    "papermill": {
     "duration": 0.058062,
     "end_time": "2020-09-08T17:54:53.380580",
     "exception": false,
     "start_time": "2020-09-08T17:54:53.322518",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "Simple Logistic Regression"
   ]
  },
  {
   "cell_type": "markdown",
   "execution_count": null,
   "metadata": {
    "papermill": {
     "duration": 0.058129,
     "end_time": "2020-09-08T17:54:53.496923",
     "exception": false,
     "start_time": "2020-09-08T17:54:53.438794",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "ilk önce logistic regression modelimizi çağrıyoruz."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 83,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2020-09-08T17:54:53.625223Z",
     "iopub.status.busy": "2020-09-08T17:54:53.624304Z",
     "iopub.status.idle": "2020-09-08T17:54:53.729631Z",
     "shell.execute_reply": "2020-09-08T17:54:53.728894Z"
    },
    "papermill": {
     "duration": 0.174277,
     "end_time": "2020-09-08T17:54:53.729754",
     "exception": false,
     "start_time": "2020-09-08T17:54:53.555477",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "ename": "ValueError",
     "evalue": "Input contains NaN, infinity or a value too large for dtype('float64').",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mValueError\u001b[0m                                Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-83-21d0f60128b4>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[0mlogreg\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mLogisticRegression\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 2\u001b[0;31m \u001b[0mlogreg\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mfit\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mX_train\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0my_train\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      3\u001b[0m \u001b[0macc_log_train\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mround\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mlogreg\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mscore\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mX_train\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0my_train\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m*\u001b[0m\u001b[0;36m100\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;36m2\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      4\u001b[0m \u001b[0macc_log_test\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mround\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mlogreg\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mscore\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mX_test\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0my_test\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m*\u001b[0m\u001b[0;36m100\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;36m2\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      5\u001b[0m \u001b[0mprint\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"Training Accuracy: % {}\"\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mformat\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0macc_log_train\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/opt/conda/lib/python3.7/site-packages/sklearn/linear_model/_logistic.py\u001b[0m in \u001b[0;36mfit\u001b[0;34m(self, X, y, sample_weight)\u001b[0m\n\u001b[1;32m   1342\u001b[0m         X, y = self._validate_data(X, y, accept_sparse='csr', dtype=_dtype,\n\u001b[1;32m   1343\u001b[0m                                    \u001b[0morder\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m\"C\"\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1344\u001b[0;31m                                    accept_large_sparse=solver != 'liblinear')\n\u001b[0m\u001b[1;32m   1345\u001b[0m         \u001b[0mcheck_classification_targets\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0my\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1346\u001b[0m         \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mclasses_\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mnp\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0munique\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0my\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/opt/conda/lib/python3.7/site-packages/sklearn/base.py\u001b[0m in \u001b[0;36m_validate_data\u001b[0;34m(self, X, y, reset, validate_separately, **check_params)\u001b[0m\n\u001b[1;32m    430\u001b[0m                 \u001b[0my\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mcheck_array\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0my\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mcheck_y_params\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    431\u001b[0m             \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 432\u001b[0;31m                 \u001b[0mX\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0my\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mcheck_X_y\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mX\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0my\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mcheck_params\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    433\u001b[0m             \u001b[0mout\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mX\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0my\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    434\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/opt/conda/lib/python3.7/site-packages/sklearn/utils/validation.py\u001b[0m in \u001b[0;36minner_f\u001b[0;34m(*args, **kwargs)\u001b[0m\n\u001b[1;32m     70\u001b[0m                           FutureWarning)\n\u001b[1;32m     71\u001b[0m         \u001b[0mkwargs\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mupdate\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m{\u001b[0m\u001b[0mk\u001b[0m\u001b[0;34m:\u001b[0m \u001b[0marg\u001b[0m \u001b[0;32mfor\u001b[0m \u001b[0mk\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0marg\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mzip\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0msig\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mparameters\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0margs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m}\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 72\u001b[0;31m         \u001b[0;32mreturn\u001b[0m \u001b[0mf\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     73\u001b[0m     \u001b[0;32mreturn\u001b[0m \u001b[0minner_f\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     74\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/opt/conda/lib/python3.7/site-packages/sklearn/utils/validation.py\u001b[0m in \u001b[0;36mcheck_X_y\u001b[0;34m(X, y, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, multi_output, ensure_min_samples, ensure_min_features, y_numeric, estimator)\u001b[0m\n\u001b[1;32m    800\u001b[0m                     \u001b[0mensure_min_samples\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mensure_min_samples\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    801\u001b[0m                     \u001b[0mensure_min_features\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mensure_min_features\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 802\u001b[0;31m                     estimator=estimator)\n\u001b[0m\u001b[1;32m    803\u001b[0m     \u001b[0;32mif\u001b[0m \u001b[0mmulti_output\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    804\u001b[0m         y = check_array(y, accept_sparse='csr', force_all_finite=True,\n",
      "\u001b[0;32m/opt/conda/lib/python3.7/site-packages/sklearn/utils/validation.py\u001b[0m in \u001b[0;36minner_f\u001b[0;34m(*args, **kwargs)\u001b[0m\n\u001b[1;32m     70\u001b[0m                           FutureWarning)\n\u001b[1;32m     71\u001b[0m         \u001b[0mkwargs\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mupdate\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m{\u001b[0m\u001b[0mk\u001b[0m\u001b[0;34m:\u001b[0m \u001b[0marg\u001b[0m \u001b[0;32mfor\u001b[0m \u001b[0mk\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0marg\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mzip\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0msig\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mparameters\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0margs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m}\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 72\u001b[0;31m         \u001b[0;32mreturn\u001b[0m \u001b[0mf\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     73\u001b[0m     \u001b[0;32mreturn\u001b[0m \u001b[0minner_f\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     74\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/opt/conda/lib/python3.7/site-packages/sklearn/utils/validation.py\u001b[0m in \u001b[0;36mcheck_array\u001b[0;34m(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, estimator)\u001b[0m\n\u001b[1;32m    643\u001b[0m         \u001b[0;32mif\u001b[0m \u001b[0mforce_all_finite\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    644\u001b[0m             _assert_all_finite(array,\n\u001b[0;32m--> 645\u001b[0;31m                                allow_nan=force_all_finite == 'allow-nan')\n\u001b[0m\u001b[1;32m    646\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    647\u001b[0m     \u001b[0;32mif\u001b[0m \u001b[0mensure_min_samples\u001b[0m \u001b[0;34m>\u001b[0m \u001b[0;36m0\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/opt/conda/lib/python3.7/site-packages/sklearn/utils/validation.py\u001b[0m in \u001b[0;36m_assert_all_finite\u001b[0;34m(X, allow_nan, msg_dtype)\u001b[0m\n\u001b[1;32m     97\u001b[0m                     \u001b[0mmsg_err\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mformat\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     98\u001b[0m                     (type_err,\n\u001b[0;32m---> 99\u001b[0;31m                      msg_dtype if msg_dtype is not None else X.dtype)\n\u001b[0m\u001b[1;32m    100\u001b[0m             )\n\u001b[1;32m    101\u001b[0m     \u001b[0;31m# for object dtype data, we only check for NaNs (GH-13254)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mValueError\u001b[0m: Input contains NaN, infinity or a value too large for dtype('float64')."
     ]
    }
   ],
   "source": [
    "logreg = LogisticRegression()\n",
    "logreg.fit(X_train, y_train)\n",
    "acc_log_train = round(logreg.score(X_train, y_train)*100,2) \n",
    "acc_log_test = round(logreg.score(X_test,y_test)*100,2)\n",
    "print(\"Training Accuracy: % {}\".format(acc_log_train))\n",
    "print(\"Testing Accuracy: % {}\".format(acc_log_test))"
   ]
  },
  {
   "cell_type": "markdown",
   "execution_count": null,
   "metadata": {
    "papermill": {
     "duration": 0.05882,
     "end_time": "2020-09-08T17:54:53.848056",
     "exception": false,
     "start_time": "2020-09-08T17:54:53.789236",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "Hyperparameter Tuning -- Grid Search -- Cross Validation\n",
    "We will compare 5 ml classifier and evaluate mean accuracy of each of them by stratified cross validation.\n",
    "\n",
    "Decision Tree\n",
    "SVM\n",
    "Random Forest\n",
    "KNN\n",
    "Logistic Regression\n"
   ]
  },
  {
   "cell_type": "markdown",
   "execution_count": null,
   "metadata": {
    "papermill": {
     "duration": 0.058164,
     "end_time": "2020-09-08T17:54:53.965316",
     "exception": false,
     "start_time": "2020-09-08T17:54:53.907152",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "bu bölümde 5 farklı ml modelini karşılaştıracağız bunlar;...\n",
    "hyperparameter tuning ile bunları karşılaştırırken aynı zamanda bu modellerin içinde bulunan parametrelerin hangisinin daha iyi olduğunu bulacağız.bunu ararken grid search yöntemini,en iyi parametere olmasını da cross validationla bulcaz\n",
    "kullanıcağımız classification yöntemleri;\n",
    "Decision Tree\n",
    "SVM\n",
    "Random Forest\n",
    "KNN\n",
    "Logistic Regression\n"
   ]
  },
  {
   "cell_type": "markdown",
   "execution_count": null,
   "metadata": {
    "papermill": {
     "duration": 0.059593,
     "end_time": "2020-09-08T17:54:54.084060",
     "exception": false,
     "start_time": "2020-09-08T17:54:54.024467",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "clasifar adında bi liste tanımlıyorum içine kullanıcağım modelleri yazıyorum.sonra hyper parameter tuning denilen şeyi yapabilmek için gerekli parametreleri seçicez"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 84,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2020-09-08T17:54:54.211342Z",
     "iopub.status.busy": "2020-09-08T17:54:54.210520Z",
     "iopub.status.idle": "2020-09-08T17:54:54.220237Z",
     "shell.execute_reply": "2020-09-08T17:54:54.219624Z"
    },
    "papermill": {
     "duration": 0.077405,
     "end_time": "2020-09-08T17:54:54.220358",
     "exception": false,
     "start_time": "2020-09-08T17:54:54.142953",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "random_state = 42\n",
    "classifier = [DecisionTreeClassifier(random_state = random_state),\n",
    "             SVC(random_state = random_state),\n",
    "             RandomForestClassifier(random_state = random_state),\n",
    "             LogisticRegression(random_state = random_state),\n",
    "             KNeighborsClassifier()]\n",
    "\n",
    "dt_param_grid = {\"min_samples_split\" : range(10,500,20),\n",
    "                \"max_depth\": range(1,20,2)}\n",
    "\n",
    "svc_param_grid = {\"kernel\" : [\"rbf\"],\n",
    "                 \"gamma\": [0.001, 0.01, 0.1, 1],\n",
    "                 \"C\": [1,10,50,100,200,300,1000]}\n",
    "\n",
    "rf_param_grid = {\"max_features\": [1,3,10],\n",
    "                \"min_samples_split\":[2,3,10],\n",
    "                \"min_samples_leaf\":[1,3,10],\n",
    "                \"bootstrap\":[False],\n",
    "                \"n_estimators\":[100,300],\n",
    "                \"criterion\":[\"gini\"]}\n",
    "\n",
    "logreg_param_grid = {\"C\":np.logspace(-3,3,7),\n",
    "                    \"penalty\": [\"l1\",\"l2\"]}\n",
    "\n",
    "knn_param_grid = {\"n_neighbors\": np.linspace(1,19,10, dtype = int).tolist(),\n",
    "                 \"weights\": [\"uniform\",\"distance\"],\n",
    "                 \"metric\":[\"euclidean\",\"manhattan\"]}\n",
    "classifier_param = [dt_param_grid,\n",
    "                   svc_param_grid,\n",
    "                   rf_param_grid,\n",
    "                   logreg_param_grid,\n",
    "                   knn_param_grid]"
   ]
  },
  {
   "cell_type": "markdown",
   "execution_count": null,
   "metadata": {
    "papermill": {
     "duration": 0.062171,
     "end_time": "2020-09-08T17:54:54.342271",
     "exception": false,
     "start_time": "2020-09-08T17:54:54.280100",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "hyper parameter dediğimiz seyler bizim mdoelilerimmi içinde kullandığımız seçilmesi gereken parametreleri seçmemize yarar\n",
    "knn de mesela n i seçmemiz gerekir\n",
    "grid search ise birinci köşede 1.e olsun ikinci köşede 3e alt satırda 1m 2m 3m böyle grid oluşturuyo daha sonra search yapıyo yani for yapısıyla yaptu-ığımızı bu metodla yapıyo.ben en iyi parametreleri ver dediğim zaman istediğim ml modelinin en iyi parametrelerini bulmuş oluyorum."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 85,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2020-09-08T17:54:54.470160Z",
     "iopub.status.busy": "2020-09-08T17:54:54.469351Z",
     "iopub.status.idle": "2020-09-08T17:55:05.067050Z",
     "shell.execute_reply": "2020-09-08T17:55:05.066070Z"
    },
    "papermill": {
     "duration": 10.665712,
     "end_time": "2020-09-08T17:55:05.067198",
     "exception": false,
     "start_time": "2020-09-08T17:54:54.401486",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Fitting 10 folds for each of 250 candidates, totalling 2500 fits\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.\n",
      "[Parallel(n_jobs=-1)]: Done  56 tasks      | elapsed:    2.4s\n",
      "[Parallel(n_jobs=-1)]: Done 2320 tasks      | elapsed:    9.7s\n",
      "[Parallel(n_jobs=-1)]: Done 2500 out of 2500 | elapsed:   10.5s finished\n"
     ]
    },
    {
     "ename": "ValueError",
     "evalue": "Input contains NaN, infinity or a value too large for dtype('float32').",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mValueError\u001b[0m                                Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-85-1a6c76cd0ae8>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m      3\u001b[0m \u001b[0;32mfor\u001b[0m \u001b[0mi\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mrange\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mlen\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mclassifier\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      4\u001b[0m     \u001b[0mclf\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mGridSearchCV\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mclassifier\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mi\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mparam_grid\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mclassifier_param\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mi\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mcv\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mStratifiedKFold\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mn_splits\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;36m10\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mscoring\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;34m\"accuracy\"\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mn_jobs\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;34m-\u001b[0m\u001b[0;36m1\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0mverbose\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;36m1\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 5\u001b[0;31m     \u001b[0mclf\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mfit\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mX_train\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0my_train\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      6\u001b[0m     \u001b[0mcv_result\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mappend\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mclf\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mbest_score_\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      7\u001b[0m     \u001b[0mbest_estimators\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mappend\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mclf\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mbest_estimator_\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/opt/conda/lib/python3.7/site-packages/sklearn/utils/validation.py\u001b[0m in \u001b[0;36minner_f\u001b[0;34m(*args, **kwargs)\u001b[0m\n\u001b[1;32m     70\u001b[0m                           FutureWarning)\n\u001b[1;32m     71\u001b[0m         \u001b[0mkwargs\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mupdate\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m{\u001b[0m\u001b[0mk\u001b[0m\u001b[0;34m:\u001b[0m \u001b[0marg\u001b[0m \u001b[0;32mfor\u001b[0m \u001b[0mk\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0marg\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mzip\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0msig\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mparameters\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0margs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m}\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 72\u001b[0;31m         \u001b[0;32mreturn\u001b[0m \u001b[0mf\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     73\u001b[0m     \u001b[0;32mreturn\u001b[0m \u001b[0minner_f\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     74\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/opt/conda/lib/python3.7/site-packages/sklearn/model_selection/_search.py\u001b[0m in \u001b[0;36mfit\u001b[0;34m(self, X, y, groups, **fit_params)\u001b[0m\n\u001b[1;32m    763\u001b[0m             \u001b[0mrefit_start_time\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mtime\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mtime\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    764\u001b[0m             \u001b[0;32mif\u001b[0m \u001b[0my\u001b[0m \u001b[0;32mis\u001b[0m \u001b[0;32mnot\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 765\u001b[0;31m                 \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mbest_estimator_\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mfit\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mX\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0my\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mfit_params\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    766\u001b[0m             \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    767\u001b[0m                 \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mbest_estimator_\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mfit\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mX\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mfit_params\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/opt/conda/lib/python3.7/site-packages/sklearn/tree/_classes.py\u001b[0m in \u001b[0;36mfit\u001b[0;34m(self, X, y, sample_weight, check_input, X_idx_sorted)\u001b[0m\n\u001b[1;32m    892\u001b[0m             \u001b[0msample_weight\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0msample_weight\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    893\u001b[0m             \u001b[0mcheck_input\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mcheck_input\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 894\u001b[0;31m             X_idx_sorted=X_idx_sorted)\n\u001b[0m\u001b[1;32m    895\u001b[0m         \u001b[0;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    896\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/opt/conda/lib/python3.7/site-packages/sklearn/tree/_classes.py\u001b[0m in \u001b[0;36mfit\u001b[0;34m(self, X, y, sample_weight, check_input, X_idx_sorted)\u001b[0m\n\u001b[1;32m    156\u001b[0m             X, y = self._validate_data(X, y,\n\u001b[1;32m    157\u001b[0m                                        validate_separately=(check_X_params,\n\u001b[0;32m--> 158\u001b[0;31m                                                             check_y_params))\n\u001b[0m\u001b[1;32m    159\u001b[0m             \u001b[0;32mif\u001b[0m \u001b[0missparse\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mX\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    160\u001b[0m                 \u001b[0mX\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0msort_indices\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/opt/conda/lib/python3.7/site-packages/sklearn/base.py\u001b[0m in \u001b[0;36m_validate_data\u001b[0;34m(self, X, y, reset, validate_separately, **check_params)\u001b[0m\n\u001b[1;32m    427\u001b[0m                 \u001b[0;31m# :(\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    428\u001b[0m                 \u001b[0mcheck_X_params\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mcheck_y_params\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mvalidate_separately\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 429\u001b[0;31m                 \u001b[0mX\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mcheck_array\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mX\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mcheck_X_params\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    430\u001b[0m                 \u001b[0my\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mcheck_array\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0my\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mcheck_y_params\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    431\u001b[0m             \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/opt/conda/lib/python3.7/site-packages/sklearn/utils/validation.py\u001b[0m in \u001b[0;36minner_f\u001b[0;34m(*args, **kwargs)\u001b[0m\n\u001b[1;32m     70\u001b[0m                           FutureWarning)\n\u001b[1;32m     71\u001b[0m         \u001b[0mkwargs\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mupdate\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m{\u001b[0m\u001b[0mk\u001b[0m\u001b[0;34m:\u001b[0m \u001b[0marg\u001b[0m \u001b[0;32mfor\u001b[0m \u001b[0mk\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0marg\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mzip\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0msig\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mparameters\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0margs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m}\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 72\u001b[0;31m         \u001b[0;32mreturn\u001b[0m \u001b[0mf\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     73\u001b[0m     \u001b[0;32mreturn\u001b[0m \u001b[0minner_f\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     74\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/opt/conda/lib/python3.7/site-packages/sklearn/utils/validation.py\u001b[0m in \u001b[0;36mcheck_array\u001b[0;34m(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, estimator)\u001b[0m\n\u001b[1;32m    643\u001b[0m         \u001b[0;32mif\u001b[0m \u001b[0mforce_all_finite\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    644\u001b[0m             _assert_all_finite(array,\n\u001b[0;32m--> 645\u001b[0;31m                                allow_nan=force_all_finite == 'allow-nan')\n\u001b[0m\u001b[1;32m    646\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    647\u001b[0m     \u001b[0;32mif\u001b[0m \u001b[0mensure_min_samples\u001b[0m \u001b[0;34m>\u001b[0m \u001b[0;36m0\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/opt/conda/lib/python3.7/site-packages/sklearn/utils/validation.py\u001b[0m in \u001b[0;36m_assert_all_finite\u001b[0;34m(X, allow_nan, msg_dtype)\u001b[0m\n\u001b[1;32m     97\u001b[0m                     \u001b[0mmsg_err\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mformat\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     98\u001b[0m                     (type_err,\n\u001b[0;32m---> 99\u001b[0;31m                      msg_dtype if msg_dtype is not None else X.dtype)\n\u001b[0m\u001b[1;32m    100\u001b[0m             )\n\u001b[1;32m    101\u001b[0m     \u001b[0;31m# for object dtype data, we only check for NaNs (GH-13254)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mValueError\u001b[0m: Input contains NaN, infinity or a value too large for dtype('float32')."
     ]
    }
   ],
   "source": [
    "cv_result = []\n",
    "best_estimators = []\n",
    "for i in range(len(classifier)):\n",
    "    clf = GridSearchCV(classifier[i], param_grid=classifier_param[i], cv = StratifiedKFold(n_splits = 10), scoring = \"accuracy\", n_jobs = -1,verbose = 1)\n",
    "    clf.fit(X_train,y_train)\n",
    "    cv_result.append(clf.best_score_)\n",
    "    best_estimators.append(clf.best_estimator_)\n",
    "    print(cv_result[i])"
   ]
  },
  {
   "cell_type": "markdown",
   "execution_count": null,
   "metadata": {
    "papermill": {
     "duration": 0.060416,
     "end_time": "2020-09-08T17:55:05.189906",
     "exception": false,
     "start_time": "2020-09-08T17:55:05.129490",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "cvresult sonuçlarımı tutucak\n",
    "best estiomr en iyileri seçicek\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 86,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2020-09-08T17:55:05.328982Z",
     "iopub.status.busy": "2020-09-08T17:55:05.327995Z",
     "iopub.status.idle": "2020-09-08T17:55:05.352226Z",
     "shell.execute_reply": "2020-09-08T17:55:05.351512Z"
    },
    "papermill": {
     "duration": 0.10162,
     "end_time": "2020-09-08T17:55:05.352352",
     "exception": false,
     "start_time": "2020-09-08T17:55:05.250732",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "ename": "ValueError",
     "evalue": "arrays must all be same length",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mValueError\u001b[0m                                Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-86-e24efe728ef9>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m      1\u001b[0m cv_results = pd.DataFrame({\"Cross Validation Means\":cv_result, \"ML Models\":[\"DecisionTreeClassifier\", \"SVM\",\"RandomForestClassifier\",\n\u001b[1;32m      2\u001b[0m              \u001b[0;34m\"LogisticRegression\"\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 3\u001b[0;31m              \"KNeighborsClassifier\"]})\n\u001b[0m\u001b[1;32m      4\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      5\u001b[0m \u001b[0mg\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0msns\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mbarplot\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"Cross Validation Means\"\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m\"ML Models\"\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mdata\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mcv_results\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/opt/conda/lib/python3.7/site-packages/pandas/core/frame.py\u001b[0m in \u001b[0;36m__init__\u001b[0;34m(self, data, index, columns, dtype, copy)\u001b[0m\n\u001b[1;32m    465\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    466\u001b[0m         \u001b[0;32melif\u001b[0m \u001b[0misinstance\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mdata\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mdict\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 467\u001b[0;31m             \u001b[0mmgr\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0minit_dict\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mdata\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mindex\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mcolumns\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mdtype\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mdtype\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    468\u001b[0m         \u001b[0;32melif\u001b[0m \u001b[0misinstance\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mdata\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mma\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mMaskedArray\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    469\u001b[0m             \u001b[0;32mimport\u001b[0m \u001b[0mnumpy\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mma\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mmrecords\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0mmrecords\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/opt/conda/lib/python3.7/site-packages/pandas/core/internals/construction.py\u001b[0m in \u001b[0;36minit_dict\u001b[0;34m(data, index, columns, dtype)\u001b[0m\n\u001b[1;32m    281\u001b[0m             \u001b[0marr\u001b[0m \u001b[0;32mif\u001b[0m \u001b[0;32mnot\u001b[0m \u001b[0mis_datetime64tz_dtype\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0marr\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;32melse\u001b[0m \u001b[0marr\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mcopy\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;32mfor\u001b[0m \u001b[0marr\u001b[0m \u001b[0;32min\u001b[0m \u001b[0marrays\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    282\u001b[0m         ]\n\u001b[0;32m--> 283\u001b[0;31m     \u001b[0;32mreturn\u001b[0m \u001b[0marrays_to_mgr\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0marrays\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mdata_names\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mindex\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mcolumns\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mdtype\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mdtype\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    284\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    285\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/opt/conda/lib/python3.7/site-packages/pandas/core/internals/construction.py\u001b[0m in \u001b[0;36marrays_to_mgr\u001b[0;34m(arrays, arr_names, index, columns, dtype, verify_integrity)\u001b[0m\n\u001b[1;32m     76\u001b[0m         \u001b[0;31m# figure out the index, if necessary\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     77\u001b[0m         \u001b[0;32mif\u001b[0m \u001b[0mindex\u001b[0m \u001b[0;32mis\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 78\u001b[0;31m             \u001b[0mindex\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mextract_index\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0marrays\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     79\u001b[0m         \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     80\u001b[0m             \u001b[0mindex\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mensure_index\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mindex\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/opt/conda/lib/python3.7/site-packages/pandas/core/internals/construction.py\u001b[0m in \u001b[0;36mextract_index\u001b[0;34m(data)\u001b[0m\n\u001b[1;32m    395\u001b[0m             \u001b[0mlengths\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mlist\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mset\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mraw_lengths\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    396\u001b[0m             \u001b[0;32mif\u001b[0m \u001b[0mlen\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mlengths\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;34m>\u001b[0m \u001b[0;36m1\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 397\u001b[0;31m                 \u001b[0;32mraise\u001b[0m \u001b[0mValueError\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"arrays must all be same length\"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    398\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    399\u001b[0m             \u001b[0;32mif\u001b[0m \u001b[0mhave_dicts\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mValueError\u001b[0m: arrays must all be same length"
     ]
    }
   ],
   "source": [
    "cv_results = pd.DataFrame({\"Cross Validation Means\":cv_result, \"ML Models\":[\"DecisionTreeClassifier\", \"SVM\",\"RandomForestClassifier\",\n",
    "             \"LogisticRegression\",\n",
    "             \"KNeighborsClassifier\"]})\n",
    "\n",
    "g = sns.barplot(\"Cross Validation Means\", \"ML Models\", data = cv_results)\n",
    "g.set_xlabel(\"Mean Accuracy\")\n",
    "g.set_title(\"Cross Validation Scores\")"
   ]
  },
  {
   "cell_type": "markdown",
   "execution_count": null,
   "metadata": {
    "papermill": {
     "duration": 0.059502,
     "end_time": "2020-09-08T17:55:05.472816",
     "exception": false,
     "start_time": "2020-09-08T17:55:05.413314",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "burda sonuçları görselleştiriyoruz daha iyi görmek için"
   ]
  },
  {
   "cell_type": "markdown",
   "execution_count": null,
   "metadata": {
    "papermill": {
     "duration": 0.060549,
     "end_time": "2020-09-08T17:55:05.599834",
     "exception": false,
     "start_time": "2020-09-08T17:55:05.539285",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "hangisinin daha başarılı olduğunu görüyoruz en iyi ml modellerini gördük"
   ]
  },
  {
   "cell_type": "markdown",
   "execution_count": null,
   "metadata": {
    "papermill": {
     "duration": 0.063422,
     "end_time": "2020-09-08T17:55:05.728661",
     "exception": false,
     "start_time": "2020-09-08T17:55:05.665239",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "Ensemble Modeling"
   ]
  },
  {
   "cell_type": "markdown",
   "execution_count": null,
   "metadata": {
    "papermill": {
     "duration": 0.062268,
     "end_time": "2020-09-08T17:55:05.854400",
     "exception": false,
     "start_time": "2020-09-08T17:55:05.792132",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "ensemble modeling yaabilmek için voting classifier kullanmak gerekir.voting classifier dediğimiz şey;bizim amacımız survived 0 mı 1 mi bikmek.voting dediği şey alex in rf ye gör 0 dfye göre 0 lr ye göre 1 ben bunların heosşnş alıp harde olarak alırsam0 lar fazla olduğu için öldü dicek ama soft olarak bakarsam olasılık olarak bakacak bu sefer olasılıkları topluyorum teker teker  oranlara bakıyorum burda da 0 çıkıyor "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 87,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2020-09-08T17:55:05.995803Z",
     "iopub.status.busy": "2020-09-08T17:55:05.993389Z",
     "iopub.status.idle": "2020-09-08T17:55:06.001323Z",
     "shell.execute_reply": "2020-09-08T17:55:06.000399Z"
    },
    "papermill": {
     "duration": 0.085334,
     "end_time": "2020-09-08T17:55:06.001514",
     "exception": false,
     "start_time": "2020-09-08T17:55:05.916180",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "ename": "IndexError",
     "evalue": "list index out of range",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mIndexError\u001b[0m                                Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-87-579da46baa22>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[0;32m----> 1\u001b[0;31m votingC = VotingClassifier(estimators = [(\"dt\",best_estimators[0]),\n\u001b[0m\u001b[1;32m      2\u001b[0m                                         \u001b[0;34m(\u001b[0m\u001b[0;34m\"rfc\"\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0mbest_estimators\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m2\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      3\u001b[0m                                         (\"lr\",best_estimators[3])],\n\u001b[1;32m      4\u001b[0m                                         voting = \"soft\", n_jobs = -1)\n\u001b[1;32m      5\u001b[0m \u001b[0mvotingC\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mvotingC\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mfit\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mX_train\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0my_train\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mIndexError\u001b[0m: list index out of range"
     ]
    }
   ],
   "source": [
    "votingC = VotingClassifier(estimators = [(\"dt\",best_estimators[0]),\n",
    "                                        (\"rfc\",best_estimators[2]),\n",
    "                                        (\"lr\",best_estimators[3])],\n",
    "                                        voting = \"soft\", n_jobs = -1)\n",
    "votingC = votingC.fit(X_train, y_train)\n",
    "print(accuracy_score(votingC.predict(X_test),y_test))"
   ]
  },
  {
   "cell_type": "markdown",
   "execution_count": null,
   "metadata": {
    "papermill": {
     "duration": 0.060466,
     "end_time": "2020-09-08T17:55:06.137833",
     "exception": false,
     "start_time": "2020-09-08T17:55:06.077367",
     "status": "completed"
    },
    "tags": []
   },
   "source": []
  },
  {
   "cell_type": "markdown",
   "execution_count": null,
   "metadata": {
    "papermill": {
     "duration": 0.059096,
     "end_time": "2020-09-08T17:55:06.257761",
     "exception": false,
     "start_time": "2020-09-08T17:55:06.198665",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "Prediction and Submission"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "papermill": {
     "duration": 0.073203,
     "end_time": "2020-09-08T17:55:06.390723",
     "exception": false,
     "start_time": "2020-09-08T17:55:06.317520",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "execution_count": null,
   "metadata": {
    "papermill": {
     "duration": 0.059362,
     "end_time": "2020-09-08T17:55:06.510680",
     "exception": false,
     "start_time": "2020-09-08T17:55:06.451318",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "kaydediğ csv formatında dışrı aktarıcaz"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 88,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2020-09-08T17:55:06.648711Z",
     "iopub.status.busy": "2020-09-08T17:55:06.647634Z",
     "iopub.status.idle": "2020-09-08T17:55:06.653112Z",
     "shell.execute_reply": "2020-09-08T17:55:06.652426Z"
    },
    "papermill": {
     "duration": 0.082475,
     "end_time": "2020-09-08T17:55:06.653235",
     "exception": false,
     "start_time": "2020-09-08T17:55:06.570760",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'votingC' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-88-5f568f2b91fd>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[0;32m----> 1\u001b[0;31m \u001b[0mtest_survived\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mpd\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mSeries\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mvotingC\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mpredict\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mtest\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mname\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;34m\"Survived\"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mastype\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mint\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      2\u001b[0m \u001b[0mresults\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mpd\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mconcat\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mtest_PassengerId\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mtest_survived\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0maxis\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;36m1\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      3\u001b[0m \u001b[0mresults\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mto_csv\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"titanic.csv\"\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mindex\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;32mFalse\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mNameError\u001b[0m: name 'votingC' is not defined"
     ]
    }
   ],
   "source": [
    "test_survived = pd.Series(votingC.predict(test), name = \"Survived\").astype(int)\n",
    "results = pd.concat([test_PassengerId, test_survived],axis = 1)\n",
    "results.to_csv(\"titanic.csv\", index = False)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.6"
  },
  "papermill": {
   "duration": 48.795766,
   "end_time": "2020-09-08T17:55:06.829692",
   "environment_variables": {},
   "exception": null,
   "input_path": "__notebook__.ipynb",
   "output_path": "__notebook__.ipynb",
   "parameters": {},
   "start_time": "2020-09-08T17:54:18.033926",
   "version": "2.1.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
