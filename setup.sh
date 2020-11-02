mkdir -p ~/.streamlit/
echo “\
[general]\n\
email = \”ya.liu1@outlook.com\”\n\
“ > ~/.streamlit/credentials.toml
echo “\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
“ > ~/.streamlit/config.toml