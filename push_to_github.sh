#!/bin/bash

# Papkaga o'tish
cd ~/Desktop/anpr_project

# Barcha o'zgarishlarni qo'shish
git add .

# O'zgarishlarni commit qilish
git commit -m "Avtomatik push: $(date)" || {
    echo "Hech qanday o'zgarish yo'q yoki commitda xato yuz berdi."
    exit 1
}

# GitHub-ga push qilish
git push origin main || {
    echo "Push qilishda xato yuz berdi. Autentifikatsiyani tekshiring yoki git pull qiling."
    exit 1
}

echo "Barcha o'zgarishlar GitHub-ga muvaffaqiyatli yuklandi!"