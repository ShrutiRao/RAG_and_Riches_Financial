from pathlib import Path

from rag_and_riches_financial.ui.chat_app import render_chat_app


def main():
    data_dir = Path(__file__).resolve().parent / "data"
    data_dir.mkdir(exist_ok=True)

    try:
        import streamlit as st
    except ImportError:
        print("Streamlit is not installed. Run `pip install -r requirements.txt` and then `streamlit run src/app.py`.")
        return

    render_chat_app(st)


if __name__ == "__main__":
    main()
