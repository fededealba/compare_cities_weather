"""Weather data logic shared by the Streamlit app and the Vercel API.

Everything in here is pure Python + pandas: no Streamlit, no filesystem writes.
The aggregation functions are lifted verbatim from the original ``app.py`` so the
numbers the site shows do not change.
"""
