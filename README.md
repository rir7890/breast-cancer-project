# breast-cancer-project
code

{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "Untitled0.ipynb",
      "provenance": [],
      "collapsed_sections": [],
      "authorship_tag": "ABX9TyNXv9ejJOVbQz91l2J9oKPy",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Vanshhhhhh/breast-cancer-prediction-/blob/main/Untitled0.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "kNrKAz_Hde8u"
      },
      "source": [
        "import pandas as pd \n",
        "import seaborn as sns "
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "imlu9QFWBJ6v"
      },
      "source": [
        "import os \n",
        "os.environ['KAGGLE_USERNAME']='vanshhgupta'\n",
        "os.environ['KAGGLE_KEY']='28e090808804004d20a5ccc3bf342b25'"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "wtmdBJu8BN0p",
        "outputId": "84fbb1c8-4d05-498c-d8a1-3667d2475b5c"
      },
      "source": [
        "! kaggle datasets download -d uciml/breast-cancer-wisconsin-data\n"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Downloading breast-cancer-wisconsin-data.zip to /content\n",
            "\r  0% 0.00/48.6k [00:00<?, ?B/s]\n",
            "\r100% 48.6k/48.6k [00:00<00:00, 18.4MB/s]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "HLs3Ll1YBVZW",
        "outputId": "19e9d74d-6634-4260-e9e0-5b6f515ae8fc"
      },
      "source": [
        "! unzip /content/breast-cancer-wisconsin-data.zip\n"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Archive:  /content/breast-cancer-wisconsin-data.zip\n",
            "  inflating: data.csv                \n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "-fyllMxWBeCX"
      },
      "source": [
        "df=pd.read_csv('/content/data.csv')"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "HhrjPnI7ggxE",
        "outputId": "35cb9db1-17bc-4f64-d461-d1fc3a791d88"
      },
      "source": [
        "df.isna().sum()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "id                           0\n",
              "diagnosis                    0\n",
              "radius_mean                  0\n",
              "texture_mean                 0\n",
              "perimeter_mean               0\n",
              "area_mean                    0\n",
              "smoothness_mean              0\n",
              "compactness_mean             0\n",
              "concavity_mean               0\n",
              "concave points_mean          0\n",
              "symmetry_mean                0\n",
              "fractal_dimension_mean       0\n",
              "radius_se                    0\n",
              "texture_se                   0\n",
              "perimeter_se                 0\n",
              "area_se                      0\n",
              "smoothness_se                0\n",
              "compactness_se               0\n",
              "concavity_se                 0\n",
              "concave points_se            0\n",
              "symmetry_se                  0\n",
              "fractal_dimension_se         0\n",
              "radius_worst                 0\n",
              "texture_worst                0\n",
              "perimeter_worst              0\n",
              "area_worst                   0\n",
              "smoothness_worst             0\n",
              "compactness_worst            0\n",
              "concavity_worst              0\n",
              "concave points_worst         0\n",
              "symmetry_worst               0\n",
              "fractal_dimension_worst      0\n",
              "Unnamed: 32                569\n",
              "dtype: int64"
            ]
          },
          "metadata": {},
          "execution_count": 10
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "LLQaEw4mgpC1"
      },
      "source": [
        ""
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "S8hjKSBoBjV3"
      },
      "source": [
        "df.dropna(axis=1,inplace=True)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "FTbgEfZ2Bp1B"
      },
      "source": [
        "from sklearn.preprocessing import LabelEncoder\n",
        "labelencoder=LabelEncoder()\n",
        "df.iloc[:,1]=labelencoder.fit_transform(df.iloc[:,1])"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "2vCSEHgBB-Sm"
      },
      "source": [
        "X=df.iloc[:,2:].values\n",
        "Y=df.iloc[:,1].values"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "iUvLHsLpCC3K"
      },
      "source": [
        "from sklearn.model_selection import train_test_split\n",
        "X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "KsE0f2NGCJ2G"
      },
      "source": [
        "from sklearn.preprocessing import StandardScaler\n",
        "sc=StandardScaler()\n",
        "X_train=sc.fit_transform(X_train)\n",
        "X_test=sc.fit_transform(X_test)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "2gAjId16CRek",
        "outputId": "e245dc01-f428-49ec-919f-624cff96255c"
      },
      "source": [
        "X_train"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([[-0.681868  , -0.27969168, -0.70692924, ..., -0.66844467,\n",
              "        -1.47925598, -0.83478537],\n",
              "       [ 0.09311454, -1.24260263,  0.05789521, ..., -0.13368387,\n",
              "        -0.49348082, -0.32967662],\n",
              "       [-0.36851766,  2.2155439 , -0.40278378, ..., -0.7694887 ,\n",
              "        -0.86079195, -0.65023726],\n",
              "       ...,\n",
              "       [ 0.0399569 , -1.36065181,  0.00314902, ..., -0.10641561,\n",
              "         0.22230496, -1.31054701],\n",
              "       [ 1.38848248,  1.67853587,  1.36937042, ...,  1.08278333,\n",
              "         0.52996727,  0.74826494],\n",
              "       [-0.16987592, -0.34913238, -0.25111658, ..., -0.96763803,\n",
              "        -0.91887106, -1.20275285]])"
            ]
          },
          "metadata": {},
          "execution_count": 16
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "qWsHZBsbCWEN",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "73db1059-dd6c-4d81-b268-c22388258394"
      },
      "source": [
        "from sklearn.linear_model import LogisticRegression\n",
        "classifier=LogisticRegression()\n",
        "classifier.fit(X_train,Y_train)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "LogisticRegression()"
            ]
          },
          "metadata": {},
          "execution_count": 17
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Y1MeXiC7Xwcd"
      },
      "source": [
        "predictions =classifier.predict(X_test)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Jn1MqnGnX5nD"
      },
      "source": [
        "from sklearn.metrics import confusion_matrix\n",
        "import seaborn as sns "
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 317
        },
        "id": "VQxxk_6LX_QH",
        "outputId": "7a8c69cc-464c-4b21-bab3-eafeeaf516b2"
      },
      "source": [
        "cm = confusion_matrix(Y_test,predictions)\n",
        "print(cm)\n",
        "sns.heatmap(cm,annot=True)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[85  5]\n",
            " [ 2 51]]\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.axes._subplots.AxesSubplot at 0x7fc378cb5ed0>"
            ]
          },
          "metadata": {},
          "execution_count": 20
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 2 Axes>"
            ]
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    }
  ]
} 
