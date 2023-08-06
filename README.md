# Lion Vs Tiger API

# Installation

Clone this repository

```bash
  git clone https://github.com/sbuDiction/lion-vs-tiger-api
```

Go to the project directory

```bash
  cd lion-vs-tiger-api
```

# Run

**Create an environment**

Create a project folder and a .venv folder within:

macOS/Linux

```bash
  python3 -m venv .wrpenv
```

Windows

```bash
  py -3 -m venv .wrpenv
```

**Activate the environment**

macOS/Linux

```bash
  . .wrpenv/bin/activate
```

Windows

```bash
  .wrpenv\Scripts\activate
```

Your shell prompt will change to show the name of the activated environment.


# Install Python Packages

Within the activated environment, use the following command to install all the required packages:


```bash
  pip install -r requirements.txt
```

<!-- web: gunicorn wsgi:app -->