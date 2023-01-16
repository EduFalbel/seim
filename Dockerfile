# Modified from https://pythonspeed.com/articles/activate-conda-dockerfile/

FROM continuumio/miniconda3

WORKDIR /app

# Create the environment:
# COPY environment.yml .
RUN conda config --add channels conda-forge
RUN conda config --set channel_priority strict
RUN conda create -n seim_clean -y python R pandas scipy matplotlib tabulate scikit-learn rpy2 r-spdep r-readr r-sf

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "seim_clean", "/bin/bash", "-c"]

# Install seim package directly from Github:
RUN echo "Installing seim:"
RUN pip install git+https://github.com/EduFalbel/seim.git

# Install spflow:
RUN echo "Installing spflow:"
RUN R -e "install.packages('spflow', repos='http://cran.r-project.org')"

# The code to run when container is started:
# COPY run.py .
# ENTRYPOINT ["conda", "activate"]