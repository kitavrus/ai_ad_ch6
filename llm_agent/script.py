#!/usr/bin/env python3
"""Точка входа CLI-чатбота (обёртка над пакетом chatbot).

Оригинальная логика перенесена в пакет chatbot/.
Запускать можно как:
    python script.py [OPTIONS]
или
    python -m chatbot.main [OPTIONS]
"""

import logging

from chatbot.main import main

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    main()
