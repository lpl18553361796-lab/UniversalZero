#!/usr/bin/env bash
set -e

cd /home/lpl/UniversalZero-main/UniversalZero-main
export RULES_LLM_BACKEND=openai
export OPENAI_BASE_URL=https://api.openai.com
export OPENAI_MODEL=gpt-4o-mini

streamlit run ui/app.py
