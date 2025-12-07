@echo off
REM Script to help upload AlzFusion to GitHub
REM Run this after creating your GitHub repository

echo ========================================
echo AlzFusion - GitHub Upload Helper
echo ========================================
echo.

REM Check if git is initialized
if not exist .git (
    echo Initializing git repository...
    git init
    echo.
)

REM Add all files
echo Adding files to git...
git add .
echo.

REM Check if there are changes to commit
git diff --cached --quiet
if %errorlevel% equ 0 (
    echo No changes to commit.
) else (
    echo Creating initial commit...
    git commit -m "Initial commit: AlzFusion - Multi-Modal Alzheimer's Prediction System"
    echo.
)

echo ========================================
echo Next Steps:
echo ========================================
echo.
echo 1. Create a repository on GitHub:
echo    https://github.com/new
echo.
echo 2. Name it: AlzFusion
echo.
echo 3. Run these commands (replace YOUR_USERNAME):
echo    git remote add origin https://github.com/YOUR_USERNAME/AlzFusion.git
echo    git branch -M main
echo    git push -u origin main
echo.
echo 4. Update README.md with your GitHub username
echo.
echo ========================================
pause

