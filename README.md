# HyDE を用いた RAG チャットボット

このリポジトリには、2 つの RAG (Retrieval Augmented Generation) チャットボットの実装が含まれています。

- **rag_basic.py:** ChromaDB ベクトルデータベースから関連するドキュメントを取得し、大規模言語モデル (LLM) を使用して応答を生成する基本的な RAG チャットボット。
- **rag_hyde.py:** HyDE (Human-in-the-Loop Data Enhancement) を利用して、取得したドキュメントと生成された応答の品質を向上させる、強化された RAG チャットボット。

## 必要なもの

- Python 3.8 以降
- OpenAI API キー (環境変数 `OPENAI_API_KEY` として設定)
- Cohere API キー (環境変数 `COHERE_API_KEY` として設定)
- LangChain API キー (環境変数 `LANGCHAIN_API_KEY` として設定)

## インストール

```bash
pip install -r requirements.txt
```

## 使い方

### 基本的な RAG チャットボット

1. 次のコマンドを実行します。`your_query` を質問に置き換えてください。

```bash
python rag_basic.py "your_query"
```

### HyDE を用いた RAG チャットボット

1. 次のコマンドを実行します。`your_query` を質問に置き換えてください。

```bash
python rag_hyde.py "your_query"
```

## 設定

次の環境変数を使用して、チャットボットを設定できます。

- `OPENAI_API_KEY`: OpenAI API キー
- `LANGCHAIN_API_KEY`: LangChain API キー
- `CHROMA_PERSIST_DIRECTORY`: ChromaDB データベースディレクトリのパス (デフォルト: `./chroma-db`)
- `CHROMA_COLLECTION_NAME`: ChromaDB コレクションの名前 (デフォルト: `wpchatbot`)

## 注意点

- チャットボットは、`gpt-4o-2024-05-13` OpenAI モデルを使用します。
- チャットボットは、パフォーマンス監視のために LangChain トレーシングを使用するように設定されています。

## 関連記事

本リポジトリについては、以下のZennの記事をご覧ください。

https://zenn.dev/khisa/articles/cc2ff969d4f2b8
