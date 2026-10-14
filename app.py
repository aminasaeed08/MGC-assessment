"""Part 4: Unified Sales Assistant Web Interface.
Runs via: python app.py -> http://127.0.0.1:5000
Features:
  1. Grounded Document Assistant with verified citations & Groq Qwen/Llama LLM
  2. Machine Learning Lead Scoring & Prioritization
  3. Multi-View Interface: Dedicated Tabs & Dedicated Routes (/assistant, /leads)
"""
import os
import json
import numpy as np
from flask import Flask, jsonify, render_template_string, request, redirect, url_for
from dotenv import load_dotenv

from assistant.qa import ask
from assistant.llm import get_llm_config, generate_llm_answer
from ml.score import get_bundle, score_lead

load_dotenv()

app = Flask(__name__)

EXAMPLES = [
    "What's the base price of a 2-bed in Block B?",
    "What's the total for a Margalla-facing corner unit on floor 15, 2-bed Block B?",
    "An overseas Pakistani buyer pays 50% upfront on a 3-Bed Executive - what's the price after discounts, and what's due at booking?",
    "What's the transfer fee?",
    "What's the rental yield on a 1-bed?",
    "Who is the anchor tenant?",
]

PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>MGC Sales Assistant · Aurora Heights</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --primary: #1e3a8a;
      --primary-gradient: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%);
      --primary-hover: #1d4ed8;
      --primary-light: #eff6ff;
      --secondary: #0f172a;
      --surface: #ffffff;
      --background: #f8fafc;
      --slate-50: #f8fafc;
      --slate-100: #f1f5f9;
      --slate-200: #e2e8f0;
      --slate-300: #cbd5e1;
      --slate-400: #94a3b8;
      --slate-500: #64748b;
      --slate-600: #475569;
      --slate-700: #334155;
      --slate-800: #1e293b;
      --slate-900: #0f172a;
      --emerald-50: #ecfdf5;
      --emerald-600: #059669;
      --emerald-700: #047857;
      --amber-50: #fffbeb;
      --amber-700: #b45309;
      --rose-50: #fff1f2;
      --rose-700: #be123c;
      --purple-50: #faf5ff;
      --purple-600: #9333ea;
      --radius-sm: 8px;
      --radius: 14px;
      --radius-lg: 18px;
      --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
      --shadow: 0 4px 16px -2px rgba(15, 23, 42, 0.06), 0 2px 6px -1px rgba(15, 23, 42, 0.03);
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif;
      background-color: var(--background);
      color: var(--slate-900);
      line-height: 1.55;
      padding: 24px 20px;
      -webkit-font-smoothing: antialiased;
    }

    .container {
      max-width: 1240px;
      margin: 0 auto;
    }

    /* Header */
    header {
      background: var(--surface);
      padding: 18px 26px;
      border-radius: var(--radius-lg);
      box-shadow: var(--shadow);
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
      margin-bottom: 20px;
      border: 1px solid rgba(226, 232, 240, 0.85);
    }

    .brand-group {
      display: flex;
      align-items: center;
      gap: 14px;
    }

    .brand-icon {
      width: 44px;
      height: 44px;
      background: var(--primary-gradient);
      border-radius: 12px;
      display: flex;
      align-items: center;
      justify-content: center;
      color: white;
      font-weight: 800;
      font-size: 1.2rem;
      box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25);
    }

    .brand-text h1 {
      font-size: 1.4rem;
      font-weight: 800;
      color: var(--slate-900);
      letter-spacing: -0.03em;
      line-height: 1.2;
    }

    .brand-text p {
      font-size: 0.8125rem;
      font-weight: 500;
      color: var(--slate-500);
    }

    .llm-badge {
      display: inline-flex;
      align-items: center;
      gap: 9px;
      padding: 7px 16px;
      border-radius: 9999px;
      font-size: 0.8125rem;
      font-weight: 600;
      background: #f0fdf4;
      border: 1px solid #bbf7d0;
      color: #15803d;
      box-shadow: var(--shadow-sm);
    }

    .llm-badge.offline {
      background: var(--slate-100);
      border-color: var(--slate-200);
      color: var(--slate-700);
    }

    .dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: #16a34a;
      box-shadow: 0 0 0 3px rgba(22, 163, 74, 0.25);
      animation: pulse 2s infinite ease-in-out;
    }

    .offline .dot {
      background: #94a3b8;
      box-shadow: none;
      animation: none;
    }

    @keyframes pulse {
      0%, 100% { opacity: 1; transform: scale(1); }
      50% { opacity: 0.5; transform: scale(1.15); }
    }

    /* Navigation Tabs */
    .nav-tabs {
      display: flex;
      gap: 8px;
      margin-bottom: 22px;
      background: var(--slate-100);
      padding: 5px;
      border-radius: 12px;
      border: 1px solid var(--slate-200);
      width: fit-content;
    }

    .tab-btn {
      padding: 8px 18px;
      border-radius: 8px;
      font-size: 0.84rem;
      font-weight: 600;
      color: var(--slate-600);
      text-decoration: none;
      transition: all 0.18s ease;
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }

    .tab-btn:hover {
      color: var(--slate-900);
    }

    .tab-btn.active {
      background: white;
      color: var(--primary);
      box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
    }

    /* Main Grid & Views */
    .grid {
      display: grid;
      grid-template-columns: 1fr;
      gap: 24px;
      align-items: start;
    }

    @media (min-width: 980px) {
      .grid-split {
        grid-template-columns: 1.15fr 0.85fr;
      }
    }

    .view-full {
      max-width: 820px;
      margin: 0 auto;
    }

    /* Card Styling */
    .card {
      background: var(--surface);
      border-radius: var(--radius-lg);
      box-shadow: var(--shadow);
      padding: 26px;
      border: 1px solid rgba(226, 232, 240, 0.9);
    }

    .card-header {
      margin-bottom: 20px;
      padding-bottom: 16px;
      border-bottom: 1px solid var(--slate-100);
      display: flex;
      align-items: center;
      gap: 12px;
    }

    .card-badge-icon {
      width: 38px;
      height: 38px;
      border-radius: 10px;
      background: var(--primary-light);
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 1.15rem;
    }

    .card-header-titles h2 {
      font-size: 1.2rem;
      font-weight: 700;
      color: var(--slate-900);
      letter-spacing: -0.02em;
    }

    .card-header-titles p {
      font-size: 0.8125rem;
      color: var(--slate-500);
    }

    /* Form Fields */
    .form-group {
      margin-bottom: 16px;
    }

    label {
      display: flex;
      justify-content: space-between;
      font-size: 0.8125rem;
      font-weight: 600;
      color: var(--slate-700);
      margin-bottom: 6px;
    }

    .label-hint {
      font-weight: 400;
      color: var(--slate-400);
      font-size: 0.75rem;
    }

    textarea, input[type="text"], select {
      width: 100%;
      padding: 10px 14px;
      border-radius: var(--radius-sm);
      border: 1.5px solid var(--slate-200);
      font-size: 0.875rem;
      font-family: inherit;
      color: var(--slate-900);
      background: var(--slate-50);
      transition: all 0.18s ease;
    }

    select {
      background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 24 24' stroke='%2364748b'%3E%3Cpath stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M19 9l-7 7-7-7'%3E%3C/path%3E%3C/svg%3E");
      background-repeat: no-repeat;
      background-position: right 12px center;
      background-size: 16px;
      padding-right: 36px;
      -webkit-appearance: none;
      cursor: pointer;
    }

    textarea:focus, input[type="text"]:focus, select:focus {
      outline: none;
      background: white;
      border-color: #3b82f6;
      box-shadow: 0 0 0 3.5px rgba(59, 130, 246, 0.12);
    }

    textarea {
      resize: vertical;
      min-height: 90px;
    }

    /* Buttons */
    .btn-primary {
      background: var(--primary-gradient);
      color: white;
      border: none;
      padding: 11px 22px;
      font-size: 0.875rem;
      font-weight: 600;
      border-radius: var(--radius-sm);
      cursor: pointer;
      box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
      transition: all 0.18s ease;
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }

    .btn-primary:hover {
      box-shadow: 0 6px 18px rgba(37, 99, 235, 0.32);
      transform: translateY(-1px);
    }

    .btn-block {
      width: 100%;
      justify-content: center;
    }

    /* Chips */
    .chips-header {
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      font-weight: 700;
      color: var(--slate-500);
      margin-top: 18px;
      margin-bottom: 10px;
      display: flex;
      align-items: center;
      gap: 6px;
    }

    .chips-grid {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 18px;
    }

    .chip-item {
      background: var(--slate-50);
      color: var(--slate-700);
      font-size: 0.8125rem;
      font-weight: 500;
      padding: 7px 12px;
      border-radius: 8px;
      text-decoration: none;
      border: 1px solid var(--slate-200);
      transition: all 0.18s ease;
    }

    .chip-item:hover {
      background: var(--primary-light);
      border-color: #93c5fd;
      color: #1e40af;
      transform: translateY(-1px);
    }

    /* Answer Container */
    .answer-container {
      margin-top: 22px;
      background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
      border: 1px solid var(--slate-200);
      border-radius: var(--radius);
      padding: 22px;
      box-shadow: var(--shadow);
    }

    .status-pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 12px;
      font-size: 0.75rem;
      font-weight: 700;
      border-radius: 9999px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }

    .status-answered { background: var(--emerald-50); color: var(--emerald-700); border: 1px solid #a7f3d0; }
    .status-conflict { background: var(--amber-50); color: var(--amber-700); border: 1px solid #fde68a; }
    .status-refused { background: var(--rose-50); color: var(--rose-700); border: 1px solid #fecdd3; }
    .status-unconfirmed { background: var(--purple-50); color: var(--purple-600); border: 1px solid #e9d5ff; }
    .status-needs_info { background: var(--primary-light); color: #1d4ed8; border: 1px solid #bfdbfe; }
    .status-not_found { background: var(--slate-100); color: var(--slate-700); border: 1px solid var(--slate-200); }

    .answer-title {
      font-size: 1.05rem;
      font-weight: 700;
      color: var(--slate-900);
      margin: 12px 0 10px 0;
      line-height: 1.4;
    }

    /* AI Synthesis Card */
    .llm-synthesis-card {
      background: linear-gradient(135deg, #fbfbfe 0%, #ffffff 100%);
      border-left: 4px solid #6366f1;
      border-radius: 10px;
      padding: 14px 18px;
      margin: 14px 0;
      border-top: 1px solid #e0e7ff;
      border-right: 1px solid #e0e7ff;
      border-bottom: 1px solid #e0e7ff;
      box-shadow: 0 2px 8px rgba(99, 102, 241, 0.06);
    }

    .llm-synthesis-header {
      font-size: 0.75rem;
      font-weight: 700;
      color: #4f46e5;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      margin-bottom: 6px;
      display: flex;
      align-items: center;
      gap: 6px;
    }

    .llm-synthesis-body {
      font-size: 0.875rem;
      color: var(--slate-800);
      line-height: 1.5;
    }

    .calc-box {
      background: #f8fafc;
      border: 1px solid var(--slate-200);
      border-radius: var(--radius-sm);
      padding: 14px 16px;
      margin: 12px 0;
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.8125rem;
      color: var(--slate-800);
      white-space: pre-wrap;
      line-height: 1.5;
    }

    .notes-group {
      margin: 14px 0;
      list-style-type: none;
    }

    .notes-group li {
      position: relative;
      padding-left: 18px;
      margin-bottom: 6px;
      font-size: 0.84rem;
      color: var(--slate-600);
    }

    .notes-group li::before {
      content: "•";
      position: absolute;
      left: 4px;
      color: var(--primary);
      font-weight: 800;
    }

    .sources-footer {
      margin-top: 14px;
      padding-top: 12px;
      border-top: 1px dashed var(--slate-200);
      font-size: 0.775rem;
      color: var(--slate-500);
    }

    .source-pill {
      display: inline-flex;
      align-items: center;
      gap: 4px;
      background: white;
      border: 1px solid var(--slate-200);
      padding: 3px 9px;
      border-radius: 6px;
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.75rem;
      color: var(--slate-700);
      margin: 3px 4px 3px 0;
    }

    /* Info Card on Left to balance height */
    .info-card {
      margin-top: 20px;
      background: var(--slate-50);
      border: 1px solid var(--slate-200);
      border-radius: var(--radius);
      padding: 18px;
    }

    .info-card-title {
      font-size: 0.8rem;
      font-weight: 700;
      color: var(--slate-600);
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 8px;
      display: flex;
      align-items: center;
      gap: 6px;
    }

    .info-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      font-size: 0.8rem;
    }

    .info-stat {
      background: white;
      border: 1px solid var(--slate-200);
      padding: 8px 12px;
      border-radius: 8px;
    }

    .info-stat-num {
      font-weight: 700;
      color: var(--primary);
      font-size: 0.95rem;
    }

    .info-stat-desc {
      color: var(--slate-500);
      font-size: 0.75rem;
    }

    /* Lead Form Rows */
    .form-row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }

    .checkbox-panel {
      background: var(--slate-50);
      padding: 14px;
      border-radius: var(--radius-sm);
      border: 1px solid var(--slate-200);
      margin: 16px 0;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }

    .checkbox-item {
      display: flex;
      align-items: center;
      gap: 10px;
      font-size: 0.8125rem;
      font-weight: 500;
      color: var(--slate-700);
      cursor: pointer;
    }

    .checkbox-item input[type="checkbox"] {
      width: 17px;
      height: 17px;
      accent-color: #2563eb;
      cursor: pointer;
    }

    /* Score Result */
    .score-banner {
      margin-top: 22px;
      background: linear-gradient(135deg, #f0fdf4 0%, #ffffff 100%);
      border: 1.5px solid #bbf7d0;
      border-radius: var(--radius);
      padding: 22px 20px;
      text-align: center;
      box-shadow: var(--shadow);
    }

    .score-banner.medium {
      background: linear-gradient(135deg, #fffbeb 0%, #ffffff 100%);
      border-color: #fde68a;
    }

    .score-banner.standard {
      background: linear-gradient(135deg, #f8fafc 0%, #ffffff 100%);
      border-color: var(--slate-200);
    }

    .score-title {
      font-size: 0.775rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--slate-500);
      margin-bottom: 6px;
    }

    .score-value {
      font-size: 3.25rem;
      font-weight: 800;
      line-height: 1;
      letter-spacing: -0.04em;
      color: #15803d;
      margin: 8px 0;
    }

    .score-banner.medium .score-value { color: #b45309; }
    .score-banner.standard .score-value { color: var(--slate-700); }

    .meter-track {
      background: var(--slate-200);
      height: 9px;
      border-radius: 9999px;
      overflow: hidden;
      margin: 14px auto 12px auto;
      max-width: 90%;
    }

    .meter-fill {
      height: 100%;
      background: linear-gradient(90deg, #10b981 0%, #059669 100%);
      border-radius: 9999px;
    }

    .score-banner.medium .meter-fill {
      background: linear-gradient(90deg, #f59e0b 0%, #d97706 100%);
    }

    .score-banner.standard .meter-fill {
      background: linear-gradient(90deg, #94a3b8 0%, #64748b 100%);
    }

    .score-summary {
      font-size: 0.8125rem;
      color: var(--slate-600);
      line-height: 1.45;
    }

    .priority-pill {
      display: inline-block;
      margin-top: 12px;
      padding: 5px 14px;
      border-radius: 9999px;
      font-weight: 700;
      font-size: 0.75rem;
      text-transform: uppercase;
    }

    .pill-high { background: #dcfce7; color: #14532d; border: 1px solid #86efac; }
    .pill-medium { background: #fef3c7; color: #78350f; border: 1px solid #fcd34d; }
    .pill-standard { background: var(--slate-100); color: var(--slate-700); border: 1px solid var(--slate-200); }

    .error-banner {
      background: var(--rose-50);
      border: 1px solid #fecdd3;
      color: var(--rose-700);
      padding: 12px 16px;
      border-radius: var(--radius-sm);
      font-size: 0.85rem;
      margin-top: 16px;
    }

    footer {
      text-align: center;
      margin-top: 36px;
      font-size: 0.775rem;
      color: var(--slate-400);
    }
  </style>
</head>
<body>

<div class="container">
  <!-- Top Header -->
  <header>
    <div class="brand-group">
      <div class="brand-icon">M</div>
      <div class="brand-text">
        <h1>MGC Sales Assistant</h1>
        <p>MGC Aurora Heights · Commercial Intelligence & Lead Prioritization</p>
      </div>
    </div>
    <div>
      {% if llm_cfg.enabled %}
        <span class="llm-badge">
          <span class="dot"></span>
          {{ llm_cfg.provider }} AI Active ({{ llm_cfg.model }})
        </span>
      {% else %}
        <span class="llm-badge offline">
          <span class="dot"></span>
          Offline Deterministic Engine
        </span>
      {% endif %}
    </div>
  </header>

  <!-- View Navigation Tabs -->
  <div class="nav-tabs">
    <a href="/?tab=split{% if question %}&q={{ question|urlencode }}{% endif %}" class="tab-btn {% if active_tab == 'split' %}active{% endif %}">
      ⚡ Unified Split View
    </a>
    <a href="/?tab=assistant{% if question %}&q={{ question|urlencode }}{% endif %}" class="tab-btn {% if active_tab == 'assistant' %}active{% endif %}">
      📄 Document Assistant
    </a>
    <a href="/?tab=leads" class="tab-btn {% if active_tab == 'leads' %}active{% endif %}">
      🎯 Lead Scorer
    </a>
  </div>

  <!-- Main Content Layout -->
  <div class="grid {% if active_tab == 'split' %}grid-split{% else %}view-full{% endif %}">

    <!-- TAB: DOCUMENT ASSISTANT -->
    {% if active_tab in ['split', 'assistant'] %}
    <div class="card">
      <div class="card-header">
        <div class="card-badge-icon">📄</div>
        <div class="card-header-titles">
          <h2>1. Ask the Documents</h2>
          <p>Grounded in Project Brochure, Price List & Booking FAQ with section citations</p>
        </div>
      </div>

      <form method="post" action="/?tab={{ active_tab }}">
        <input type="hidden" name="form" value="ask">
        <div class="form-group">
          <label for="question">
            <span>Enter your question</span>
            <span class="label-hint">Plain language supported</span>
          </label>
          <textarea id="question" name="question" placeholder="e.g. What's the total for a Margalla-facing corner unit on floor 15, 2-bed Block B?">{{ question }}</textarea>
        </div>
        <button type="submit" class="btn-primary">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M5 12h14M12 5l7 7-7 7"/></svg>
          Ask Assistant
        </button>
      </form>

      <div class="chips-header">
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/></svg>
        Quick Example Questions
      </div>
      <div class="chips-grid">
        {% for ex in examples %}
          <a class="chip-item" href="/?tab={{ active_tab }}&q={{ ex|urlencode }}">💬 {{ ex }}</a>
        {% endfor %}
      </div>

      {% if answer %}
        <div class="answer-container">
          <span class="status-pill status-{{ answer.status }}">
            {% if answer.status == 'answered' %}🟢 Grounded Answer
            {% elif answer.status == 'conflict' %}⚠️ Document Conflict Flagged
            {% elif answer.status == 'refused' %}🛑 Refused by Policy
            {% elif answer.status == 'unconfirmed' %}⏳ Status Unconfirmed
            {% elif answer.status == 'needs_info' %}ℹ️ Additional Info Required
            {% else %}⚪ Not in Documents
            {% endif %}
          </span>

          <div class="answer-title">{{ answer.headline }}</div>

          {% if llm_text %}
            <div class="llm-synthesis-card">
              <div class="llm-synthesis-header">
                ✨ AI Assistant Synthesis ({{ llm_cfg.provider }} · {{ llm_cfg.model }})
              </div>
              <div class="llm-synthesis-body">{{ llm_text }}</div>
            </div>
          {% endif %}

          {% if answer.details %}
            <div class="calc-box">{% for line in answer.details %}{{ line }}
{% endfor %}</div>
          {% endif %}

          {% if answer.notes %}
            <ul class="notes-group">
              {% for note in answer.notes %}
                <li>{{ note }}</li>
              {% endfor %}
            </ul>
          {% endif %}

          {% if answer.sources %}
            <div class="sources-footer">
              <strong style="color: var(--slate-700);">Verified Sources:</strong><br>
              {% for src in answer.sources %}
                <span class="source-pill">📄 {{ src }}</span>
              {% endfor %}
            </div>
          {% endif %}
        </div>
      {% endif %}

      <!-- Project Quick Stats to balance height nicely -->
      {% if active_tab == 'split' and not answer %}
        <div class="info-card">
          <div class="info-card-title">
            <span>🏢 MGC Aurora Heights Quick Guide</span>
          </div>
          <div class="info-grid">
            <div class="info-stat">
              <div class="info-stat-num">22 Storeys</div>
              <div class="info-stat-desc">Residential & Commercial Podium</div>
            </div>
            <div class="info-stat">
              <div class="info-stat-num">418 Units</div>
              <div class="info-stat-desc">Block A (Q3 2027) & Block B</div>
            </div>
            <div class="info-stat">
              <div class="info-stat-num">4-Year Plan</div>
              <div class="info-stat-desc">10% Token + 16 Quarterly Installments</div>
            </div>
            <div class="info-stat">
              <div class="info-stat-num">GT Road, ISB</div>
              <div class="info-stat-desc">4 km from Al-Jannat Mall</div>
            </div>
          </div>
        </div>
      {% endif %}
    </div>
    {% endif %}

    <!-- TAB: LEAD SCORING -->
    {% if active_tab in ['split', 'leads'] %}
    <div class="card">
      <div class="card-header">
        <div class="card-badge-icon">🎯</div>
        <div class="card-header-titles">
          <h2>2. Score a Lead</h2>
          <p>Predict conversion probability based on 9,000+ historical CRM records</p>
        </div>
      </div>

      <form method="post" action="/?tab={{ active_tab }}">
        <input type="hidden" name="form" value="score">
        
        <div class="form-row">
          <div class="form-group">
            <label>Budget (PKR lac) *</label>
            <input type="text" name="budget_pkr_lac" value="{{ f.budget_pkr_lac }}" placeholder="e.g. 150">
          </div>
          <div class="form-group">
            <label>City</label>
            <select name="city">
              {% for c in choices.city %}
                <option value="{{ c }}" {{ 'selected' if c==f.city }}>{{ c }}</option>
              {% endfor %}
            </select>
          </div>
        </div>

        <div class="form-row">
          <div class="form-group">
            <label>Lead Source</label>
            <select name="source">
              {% for s in choices.source %}
                <option value="{{ s }}" {{ 'selected' if s==f.source }}>{{ s }}</option>
              {% endfor %}
            </select>
          </div>
          <div class="form-group">
            <label>Property Type</label>
            <select name="property_type">
              {% for pt in choices.property_type %}
                <option value="{{ pt }}" {{ 'selected' if pt==f.property_type }}>{{ pt }}</option>
              {% endfor %}
            </select>
          </div>
        </div>

        <div class="form-row">
          <div class="form-group">
            <label>Site Visits</label>
            <input type="text" name="site_visits" value="{{ f.site_visits }}">
          </div>
          <div class="form-group">
            <label>Calls Made</label>
            <input type="text" name="calls_made" value="{{ f.calls_made }}">
          </div>
        </div>

        <div class="form-row">
          <div class="form-group">
            <label>WhatsApp Replies</label>
            <input type="text" name="whatsapp_replies" value="{{ f.whatsapp_replies }}">
          </div>
          <div class="form-group">
            <label>Call Duration (Sec)</label>
            <input type="text" name="total_call_seconds" value="{{ f.total_call_seconds }}">
          </div>
        </div>

        <div class="form-row">
          <div class="form-group">
            <label>
              <span>First Response</span>
              <span class="label-hint">minutes</span>
            </label>
            <input type="text" name="first_response_minutes" value="{{ f.first_response_minutes }}" placeholder="optional">
          </div>
          <div class="form-group">
            <label>Bedrooms</label>
            <input type="text" name="bedrooms" value="{{ f.bedrooms }}" placeholder="optional">
          </div>
        </div>

        <div class="form-group">
          <label>Agent Experience (Years)</label>
          <input type="text" name="agent_experience_years" value="{{ f.agent_experience_years }}" placeholder="optional">
        </div>

        <div class="checkbox-panel">
          <label class="checkbox-item">
            <input type="checkbox" name="is_overseas" {{ 'checked' if f.is_overseas }}>
            <span>Overseas Pakistani Buyer (Roshan Digital Account)</span>
          </label>
          <label class="checkbox-item">
            <input type="checkbox" name="referred_by_existing_client" {{ 'checked' if f.referred_by_existing_client }}>
            <span>Referred by Existing Client</span>
          </label>
          <label class="checkbox-item">
            <input type="checkbox" name="has_financing_approved" {{ 'checked' if f.has_financing_approved }}>
            <span>Bank Financing Approved</span>
          </label>
        </div>

        <button type="submit" class="btn-primary btn-block">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
          Calculate Lead Score
        </button>
      </form>

      {% if result %}
        <div class="score-banner {% if result.top_percent_of_recent_leads <= 15 %}high{% elif result.top_percent_of_recent_leads <= 35 %}medium{% else %}standard{% endif %}">
          <div class="score-title">Predicted Conversion Probability</div>
          <div class="score-value">{{ result.percent }}%</div>

          <div class="meter-track">
            <div class="meter-fill" style="width: {{ [result.percent, 100]|min }}%;"></div>
          </div>

          <div class="score-summary">
            Ranks in the top <strong>{{ result.top_percent_of_recent_leads }}%</strong> of recent incoming leads.
            <br>
            Average CRM conversion base rate is <strong>{{ base_rate }}%</strong>.
          </div>

          {% if result.top_percent_of_recent_leads <= 15 %}
            <span class="priority-pill pill-high">🔥 High Priority — Call Immediately</span>
          {% elif result.top_percent_of_recent_leads <= 35 %}
            <span class="priority-pill pill-medium">⚡ Medium Priority — Follow Up Soon</span>
          {% else %}
            <span class="priority-pill pill-standard">📞 Standard Priority Lead</span>
          {% endif %}
        </div>
      {% endif %}

      {% if error %}
        <div class="error-banner">
          ⚠️ {{ error }}
        </div>
      {% endif %}
    </div>
    {% endif %}

  </div>

  <footer>
    MGC Aurora Heights · GT Road, Islamabad · AI Developer & Engineer Assessment Build
  </footer>
</div>

</body>
</html>
"""

FLOATS = [
    "budget_pkr_lac", "bedrooms", "first_response_minutes", "calls_made",
    "total_call_seconds", "whatsapp_replies", "site_visits", "agent_experience_years"
]
FLAGS = ["is_overseas", "referred_by_existing_client", "has_financing_approved"]
DEFAULTS = {
    "budget_pkr_lac": "150", "city": "Islamabad", "source": "Facebook Ads",
    "property_type": "Apartment", "site_visits": "0", "calls_made": "0",
    "whatsapp_replies": "0", "total_call_seconds": "0", "first_response_minutes": "",
    "bedrooms": "", "agent_experience_years": "", "is_overseas": False,
    "referred_by_existing_client": False, "has_financing_approved": False
}


def parse_lead(form) -> dict:
    """Form strings -> numbers. Blank optional fields become NaN (the model imputes them)."""
    lead = {
        "city": form.get("city"),
        "source": form.get("source"),
        "property_type": form.get("property_type")
    }
    for k in FLOATS:
        raw = form.get(k)
        v = "" if raw is None else str(raw).strip()
        lead[k] = float(v) if v else np.nan
        if v and lead[k] < 0:
            raise ValueError(f"{k} cannot be negative")
    for k in FLAGS:
        lead[k] = 1 if form.get(k) else 0
    if np.isnan(lead["budget_pkr_lac"]) or lead["budget_pkr_lac"] <= 0:
        raise ValueError("Please enter a budget greater than 0 (PKR lac).")
    return lead


@app.route("/favicon.ico")
def favicon():
    """Silences browser 404 logs cleanly."""
    return ("", 204)


@app.route("/", methods=["GET", "POST"])
def index():
    bundle = get_bundle()
    llm_cfg = get_llm_config()
    active_tab = request.args.get("tab", "split")
    if active_tab not in ["split", "assistant", "leads"]:
        active_tab = "split"

    ctx = dict(
        examples=EXAMPLES,
        choices=bundle["choices"],
        question="",
        answer=None,
        llm_text=None,
        llm_cfg=llm_cfg,
        result=None,
        error=None,
        active_tab=active_tab,
        f=dict(DEFAULTS),
        base_rate=round(100 * float(np.mean(bundle["ref_scores"])), 1)
    )

    if request.method == "GET" and request.args.get("q"):
        ctx["question"] = request.args["q"]
        ctx["answer"] = ask(ctx["question"])
        if llm_cfg["enabled"]:
            ctx["llm_text"] = generate_llm_answer(ctx["question"], ctx["answer"])
    elif request.method == "POST":
        if request.form.get("form") == "ask":
            ctx["question"] = request.form.get("question", "")
            ctx["answer"] = ask(ctx["question"])
            if llm_cfg["enabled"]:
                ctx["llm_text"] = generate_llm_answer(ctx["question"], ctx["answer"])
        else:
            ctx["f"] = {
                k: (request.form.get(k) if k not in FLAGS else bool(request.form.get(k)))
                for k in DEFAULTS
            }
            try:
                ctx["result"] = score_lead(parse_lead(request.form))
            except ValueError as e:
                ctx["error"] = str(e)
    return render_template_string(PAGE, **ctx)


@app.route("/assistant", methods=["GET", "POST"])
def assistant_page():
    """Dedicated full-page Document Assistant route."""
    return redirect(url_for("index", tab="assistant", **request.args))


@app.route("/leads", methods=["GET", "POST"])
def leads_page():
    """Dedicated full-page Lead Scoring route."""
    return redirect(url_for("index", tab="leads", **request.args))


@app.post("/api/ask")
def api_ask():
    question = (request.get_json(silent=True) or {}).get("question", "")
    a = ask(question)
    llm_cfg = get_llm_config()
    llm_text = generate_llm_answer(question, a) if llm_cfg["enabled"] else None
    return jsonify(
        status=a.status,
        answer=a.text(),
        sources=a.sources,
        llm_answer=llm_text
    )


@app.post("/api/score")
def api_score():
    try:
        return jsonify(score_lead(parse_lead(request.get_json(silent=True) or {})))
    except (ValueError, TypeError) as e:
        return jsonify(error=str(e)), 400


if __name__ == "__main__":
    get_bundle()
    app.run(debug=False)
