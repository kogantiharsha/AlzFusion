# Contributing to AlzFusion

Thank you for your interest in contributing to AlzFusion! This document provides guidelines and instructions for contributing.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:
- Clear description of the bug
- Steps to reproduce
- Expected vs actual behavior
- System information (OS, Python version, etc.)

### Suggesting Enhancements

We welcome suggestions for new features or improvements! Please open an issue with:
- Clear description of the enhancement
- Use case and benefits
- Possible implementation approach (if you have ideas)

### Code Contributions

1. **Fork the repository**
2. **Create a branch** for your feature/fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes** following the code style
4. **Test your changes**:
   ```bash
   python -m pytest tests/  # If tests exist
   python example_usage.py   # Test with example
   ```
5. **Commit your changes**:
   ```bash
   git commit -m "Add: Description of your changes"
   ```
6. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```
7. **Open a Pull Request**

## Code Style

- Follow PEP 8 Python style guide
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Keep functions focused and modular
- Add comments for complex logic

## Project Structure

- `src/models/` - Model architectures
- `src/data/` - Data loading and preprocessing
- `src/training/` - Training utilities
- `src/utils/` - Utility functions
- `scripts/` - Executable scripts

## Testing

Before submitting, please:
- Test your changes locally
- Ensure imports work correctly
- Run the example script if applicable
- Check for any linting errors

## Questions?

Feel free to open an issue for any questions or discussions!

Thank you for contributing to AlzFusion! 🚀

