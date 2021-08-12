FROM python:3.8
EXPOSE 80
WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt
COPY . .
CMD streamlit run scripts/app.py

# ENTRYPOINT ["server.port", "80", \
# "--server.enableCORS", "true", \
# "--browser.serverAddress", "0.0.0.0", \
# "--browser.serverPort", "443"]