# Contributing to optpricing

Thank you for your interest in improving optpricing! Even though this is a small project, contributions are welcome—bug reports, documentation fixes, or new feature suggestions.

## Getting Started

1. **Fork** the repository on GitHub.
2. **Clone** your fork locally:

   ```bash
   git clone https://github.com/diljit22/quantfin.git
   cd optpricing
   ```

3. **Install** development dependencies:

   ```bash
   pip install -e '.[app,dev]'
   ```

## Coding Style

* This project uses **Ruff** for linting and formatting:

  ```bash
  ruff format .
  ruff check .
  ```
  
* Please write clear, modular code and include docstrings in NumPy style.

## Testing

Run the full test suite with:

```bash
pytest
```

> **Note:** Some dashboard-related files/functions do not yet have pytest coverage.

## Submitting Changes

1. Create a **feature branch** (`git checkout -b feature/my-change`).
2. Commit your changes with descriptive messages.
3. Push to your fork (`git push origin feature/my-change`).
4. Open a **Pull Request** against `main`.

Please reference any relevant issue numbers in your PR description.

## Reporting Issues and Requests

* For bugs, use the **Bug report** issue template.
* For new features, use the **Feature request** template.

---

For more details, see our [Code of Conduct](CODE_OF_CONDUCT.md) and [Pull Request Template](PULL_REQUEST_TEMPLATE.md).
