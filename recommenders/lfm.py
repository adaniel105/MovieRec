{
  "cells": [
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "oa2enTrgI6Wd",
        "outputId": "7c0aeeea-cc98-470e-f25b-e8eee2d9e51d"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Collecting LightFM\n",
            "  Downloading lightfm-1.17.tar.gz (316 kB)\n",
            "\u001b[?25l     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m0.0/316.4 kB\u001b[0m \u001b[31m?\u001b[0m eta \u001b[36m-:--:--\u001b[0m\r\u001b[2K     \u001b[91m━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[90m╺\u001b[0m\u001b[90m━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m184.3/316.4 kB\u001b[0m \u001b[31m5.3 MB/s\u001b[0m eta \u001b[36m0:00:01\u001b[0m\r\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m316.4/316.4 kB\u001b[0m \u001b[31m5.2 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25h  Preparing metadata (setup.py) ... \u001b[?25l\u001b[?25hdone\n",
            "Requirement already satisfied: numpy in /usr/local/lib/python3.10/dist-packages (from LightFM) (1.26.4)\n",
            "Requirement already satisfied: scipy>=0.17.0 in /usr/local/lib/python3.10/dist-packages (from LightFM) (1.13.1)\n",
            "Requirement already satisfied: requests in /usr/local/lib/python3.10/dist-packages (from LightFM) (2.32.3)\n",
            "Requirement already satisfied: scikit-learn in /usr/local/lib/python3.10/dist-packages (from LightFM) (1.5.2)\n",
            "Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests->LightFM) (3.3.2)\n",
            "Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests->LightFM) (3.10)\n",
            "Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests->LightFM) (2.2.3)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests->LightFM) (2024.8.30)\n",
            "Requirement already satisfied: joblib>=1.2.0 in /usr/local/lib/python3.10/dist-packages (from scikit-learn->LightFM) (1.4.2)\n",
            "Requirement already satisfied: threadpoolctl>=3.1.0 in /usr/local/lib/python3.10/dist-packages (from scikit-learn->LightFM) (3.5.0)\n",
            "Building wheels for collected packages: LightFM\n",
            "  Building wheel for LightFM (setup.py) ... \u001b[?25l\u001b[?25hdone\n",
            "  Created wheel for LightFM: filename=lightfm-1.17-cp310-cp310-linux_x86_64.whl size=808331 sha256=b6d699aa262e73549bf1bc4098c1f8f73b64dc7224a583d4b278700c313b380d\n",
            "  Stored in directory: /root/.cache/pip/wheels/4f/9b/7e/0b256f2168511d8fa4dae4fae0200fdbd729eb424a912ad636\n",
            "Successfully built LightFM\n",
            "Installing collected packages: LightFM\n",
            "Successfully installed LightFM-1.17\n"
          ]
        }
      ],
      "source": [
        "!pip install LightFM"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "6743jtOIIxJe",
        "outputId": "01c5fe4c-b020-4d0c-a478-afee71f21ef2"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Precision: train 0.35, test 0.04.\n",
            "AUC: train 0.97, test 0.92.\n"
          ]
        }
      ],
      "source": [
        "from lightfm import LightFM\n",
        "from lightfm.datasets import fetch_movielens\n",
        "from lightfm.evaluation import precision_at_k, auc_score\n",
        "\n",
        "data = fetch_movielens(min_rating=5.0)\n",
        "train = data['train']\n",
        "test = data['test']\n",
        "\n",
        "model = LightFM(learning_rate=0.05,loss=\"warp\")\n",
        "model.fit(train, epochs=30, num_threads=2)\n",
        "\n",
        "train_precision = precision_at_k(model, train, k=10).mean()\n",
        "test_precision = precision_at_k(model, test, k=10).mean()\n",
        "train_auc = auc_score(model, train).mean()\n",
        "test_auc = auc_score(model, test).mean()\n",
        "\n",
        "print('Precision: train %.2f, test %.2f.' % (train_precision, test_precision))\n",
        "print('AUC: train %.2f, test %.2f.' % (train_auc, test_auc))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "NVsW4l91Kdm0",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "4001767a-ad74-46ae-f512-3ee03a096616"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([-0.76900125, -1.520023  , -0.6437492 , -1.4554318 , -0.9598947 ],\n",
              "      dtype=float32)"
            ]
          },
          "metadata": {},
          "execution_count": 3
        }
      ],
      "source": [
        "import pandas as pd\n",
        "score = model.predict(1, [47,50, 223, 235,349]) #takes in user and item id, optionally other features such as item and user features.\n",
        "ratings = pd.read_csv(\"/content/ratings.csv\")\n",
        "score"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "2NluKm04kVgy"
      },
      "outputs": [],
      "source": [
        "sample = pd.DataFrame({\n",
        "    \"userId\" : 1, \"itemId\" : [47, 50,223,235,349]\n",
        "})\n",
        "sample[\"scores\"] = score"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "id": "OeDbgiW6k1ty",
        "outputId": "c129e7ae-bdb0-4cf2-c4ad-14d24d59019a"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "   userId  itemId    scores\n",
              "2       1     223 -0.643749\n",
              "0       1      47 -0.769001\n",
              "4       1     349 -0.959895\n",
              "3       1     235 -1.455432\n",
              "1       1      50 -1.520023"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-920404f9-5446-4bf8-965b-d0e5cabb3238\" class=\"colab-df-container\">\n",
              "    <div>\n",
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
              "      <th>userId</th>\n",
              "      <th>itemId</th>\n",
              "      <th>scores</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>1</td>\n",
              "      <td>223</td>\n",
              "      <td>-0.643749</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>1</td>\n",
              "      <td>47</td>\n",
              "      <td>-0.769001</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>1</td>\n",
              "      <td>349</td>\n",
              "      <td>-0.959895</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>1</td>\n",
              "      <td>235</td>\n",
              "      <td>-1.455432</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>1</td>\n",
              "      <td>50</td>\n",
              "      <td>-1.520023</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "    <div class=\"colab-df-buttons\">\n",
              "\n",
              "  <div class=\"colab-df-container\">\n",
              "    <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-920404f9-5446-4bf8-965b-d0e5cabb3238')\"\n",
              "            title=\"Convert this dataframe to an interactive table.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\" viewBox=\"0 -960 960 960\">\n",
              "    <path d=\"M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "\n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    .colab-df-buttons div {\n",
              "      margin-bottom: 4px;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "    <script>\n",
              "      const buttonEl =\n",
              "        document.querySelector('#df-920404f9-5446-4bf8-965b-d0e5cabb3238 button.colab-df-convert');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      async function convertToInteractive(key) {\n",
              "        const element = document.querySelector('#df-920404f9-5446-4bf8-965b-d0e5cabb3238');\n",
              "        const dataTable =\n",
              "          await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                    [key], {});\n",
              "        if (!dataTable) return;\n",
              "\n",
              "        const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "          '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "          + ' to learn more about interactive tables.';\n",
              "        element.innerHTML = '';\n",
              "        dataTable['output_type'] = 'display_data';\n",
              "        await google.colab.output.renderOutput(dataTable, element);\n",
              "        const docLink = document.createElement('div');\n",
              "        docLink.innerHTML = docLinkHtml;\n",
              "        element.appendChild(docLink);\n",
              "      }\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "\n",
              "<div id=\"df-34ac9b42-52d5-44e8-9625-b16e6d8b1c14\">\n",
              "  <button class=\"colab-df-quickchart\" onclick=\"quickchart('df-34ac9b42-52d5-44e8-9625-b16e6d8b1c14')\"\n",
              "            title=\"Suggest charts\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "<svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "     width=\"24px\">\n",
              "    <g>\n",
              "        <path d=\"M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z\"/>\n",
              "    </g>\n",
              "</svg>\n",
              "  </button>\n",
              "\n",
              "<style>\n",
              "  .colab-df-quickchart {\n",
              "      --bg-color: #E8F0FE;\n",
              "      --fill-color: #1967D2;\n",
              "      --hover-bg-color: #E2EBFA;\n",
              "      --hover-fill-color: #174EA6;\n",
              "      --disabled-fill-color: #AAA;\n",
              "      --disabled-bg-color: #DDD;\n",
              "  }\n",
              "\n",
              "  [theme=dark] .colab-df-quickchart {\n",
              "      --bg-color: #3B4455;\n",
              "      --fill-color: #D2E3FC;\n",
              "      --hover-bg-color: #434B5C;\n",
              "      --hover-fill-color: #FFFFFF;\n",
              "      --disabled-bg-color: #3B4455;\n",
              "      --disabled-fill-color: #666;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart {\n",
              "    background-color: var(--bg-color);\n",
              "    border: none;\n",
              "    border-radius: 50%;\n",
              "    cursor: pointer;\n",
              "    display: none;\n",
              "    fill: var(--fill-color);\n",
              "    height: 32px;\n",
              "    padding: 0;\n",
              "    width: 32px;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart:hover {\n",
              "    background-color: var(--hover-bg-color);\n",
              "    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "    fill: var(--button-hover-fill-color);\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart-complete:disabled,\n",
              "  .colab-df-quickchart-complete:disabled:hover {\n",
              "    background-color: var(--disabled-bg-color);\n",
              "    fill: var(--disabled-fill-color);\n",
              "    box-shadow: none;\n",
              "  }\n",
              "\n",
              "  .colab-df-spinner {\n",
              "    border: 2px solid var(--fill-color);\n",
              "    border-color: transparent;\n",
              "    border-bottom-color: var(--fill-color);\n",
              "    animation:\n",
              "      spin 1s steps(1) infinite;\n",
              "  }\n",
              "\n",
              "  @keyframes spin {\n",
              "    0% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "      border-left-color: var(--fill-color);\n",
              "    }\n",
              "    20% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    30% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    40% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    60% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    80% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "    90% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "  }\n",
              "</style>\n",
              "\n",
              "  <script>\n",
              "    async function quickchart(key) {\n",
              "      const quickchartButtonEl =\n",
              "        document.querySelector('#' + key + ' button');\n",
              "      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.\n",
              "      quickchartButtonEl.classList.add('colab-df-spinner');\n",
              "      try {\n",
              "        const charts = await google.colab.kernel.invokeFunction(\n",
              "            'suggestCharts', [key], {});\n",
              "      } catch (error) {\n",
              "        console.error('Error during call to suggestCharts:', error);\n",
              "      }\n",
              "      quickchartButtonEl.classList.remove('colab-df-spinner');\n",
              "      quickchartButtonEl.classList.add('colab-df-quickchart-complete');\n",
              "    }\n",
              "    (() => {\n",
              "      let quickchartButtonEl =\n",
              "        document.querySelector('#df-34ac9b42-52d5-44e8-9625-b16e6d8b1c14 button');\n",
              "      quickchartButtonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "    })();\n",
              "  </script>\n",
              "</div>\n",
              "\n",
              "    </div>\n",
              "  </div>\n"
            ],
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "dataframe",
              "summary": "{\n  \"name\": \"sample\",\n  \"rows\": 5,\n  \"fields\": [\n    {\n      \"column\": \"userId\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0,\n        \"min\": 1,\n        \"max\": 1,\n        \"num_unique_values\": 1,\n        \"samples\": [\n          1\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"itemId\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 130,\n        \"min\": 47,\n        \"max\": 349,\n        \"num_unique_values\": 5,\n        \"samples\": [\n          47\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"scores\",\n      \"properties\": {\n        \"dtype\": \"float32\",\n        \"num_unique_values\": 5,\n        \"samples\": [\n          -0.7690012454986572\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    }\n  ]\n}"
            }
          },
          "metadata": {},
          "execution_count": 5
        }
      ],
      "source": [
        "sample.sort_values(by='scores' ,ascending=False)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "oI1Duwe_lpeZ"
      },
      "outputs": [],
      "source": [
        "movies = pd.read_csv(\"/content/movies.csv\")\n",
        "def get_recs(df):\n",
        "  recs = [movies[\"title\"].loc[movies[\"movieId\"] == i] for i in sample[\"itemId\"]]\n",
        "  hold = []\n",
        "  for i in range(len(recs)):\n",
        "    hold.append(recs[i].tolist()[0].strip())\n",
        "  return hold"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "hold = get_recs(movies)\n",
        "hold.pop(0)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 35
        },
        "id": "6BXE4E4RwR10",
        "outputId": "67453c6c-902a-4c60-c74b-376b9e43d20f"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "'Seven (a.k.a. Se7en) (1995)'"
            ],
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "string"
            }
          },
          "metadata": {},
          "execution_count": 10
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "item_features = data['item_features']\n",
        "movie_labels = data[\"item_feature_labels\"]"
      ],
      "metadata": {
        "id": "djCvhbHC2b0W"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "movie_labels.size"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "ks-DU2K62o5R",
        "outputId": "d5768824-23c4-462f-bd12-2d313be30b2b"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "1682"
            ]
          },
          "metadata": {},
          "execution_count": 21
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import numpy as np\n",
        "def get_similar_movies(model, movieId):\n",
        "  movie_embed = (model.item_embeddings.T\n",
        "                 / np.linalg.norm(model.item_embeddings, axis=1)).T\n",
        "\n",
        "  query_embed = movie_embed[movieId]\n",
        "  similarity = np.dot(movie_embed, query_embed)\n",
        "  most_similar = np.argsort(-similarity)[1:4]\n",
        "\n",
        "  return most_similar\n",
        "\n",
        "movie = \"Jumanji\"\n",
        "movieId = movie_labels.tolist().index(movie[0])\n",
        "print(f\"Most similar movies for {movie_labels[movieId]}: {movie_labels[get_similar_movies(model, movieId)]}\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "w9Z2Aqx3zjco",
        "outputId": "62d40e52-4b25-4719-dc91-c7b235e76699"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Most similar movies for J: ['S' 'H' 'A']\n"
          ]
        }
      ]
    }
  ],
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyNZ4AhmvBoiQ22/DwPBBS/r"
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}