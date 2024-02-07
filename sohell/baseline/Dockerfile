
FROM python:3.11.3-slim

WORKDIR /app

RUN apt-get update  \
    && apt-get install -y  \
    g++  \
    swig \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# psycopg2 is only needed as long as we use find_cycles from simulator/find_cycles.py
RUN pip install \
    psycopg2-binary \
    typed-argument-parser \
    tqdm \
    ConfigSpace \
    smac \
    matplotlib \
    "attrs>=21" \
    matplotlib \
    pandas

COPY ./simulator/cpp /app/simulator/cpp

RUN pip install ./simulator/cpp

# . is src folder
COPY ./learning/bayesian_regression /app/learning/bayesian_regression
COPY ./simulator /app/simulator
COPY ./plotting /app/plotting
# soc_ocv.txt needs to be in working directory
COPY ./simulator/soc_ocv.txt /app/

ENV PYTHONPATH=/app/learning/bayesian_regression:/app/simulator:/app/plotting:${PYTHONPATH}
ENV PYTHONUNBUFFERED=1

CMD [\
        "python",\
        "-m labeling",\
        "--sohc_range",\
        "40",\
        "90",\
        "-d",\
        "learning/bayesian_regression",\
        "--n_trials",\
        "10"\
    ]