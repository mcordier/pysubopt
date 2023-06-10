# PySubOpt

PySubOpt is a Python project monorepo that provides submodular optimization packages for various applications. This monorepo includes the following packages:

- **opt-submodular**: This is the core code library for submodular optimization. It contains implementation of various submodular optimization algorithms and functions.
- **doc-summarize**: This package utilizes the `opt-submodular` library to perform text summarization. It provides functionalities to summarize texts using submodular optimization techniques.
- **opt-network**: This package focuses on maximizing network contamination using submodular optimization. It utilizes the `opt-submodular` library to identify the optimal set of nodes to maximize contamination in a network.

## Setup

To install the PySubOpt packages, please follow the instructions below:

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/your-username/PySubOpt.git
   ```

2. Change into the project directory:

   ```bash
   cd PySubOpt
   ```

3. Run the setup command to install the required dependencies:

   ```bash
   make setup
   ```

This will install all the necessary dependencies for the packages.

## Quickstart

To get started with each package, refer to the README file of each respective package:

- **opt-submodular**: Refer to the README file in the `opt-submodular` directory for detailed instructions on how to use the core library for submodular optimization.

- **doc-summarize**: Refer to the README file in the `doc-summarize` directory for examples and guidelines on how to use the package for text summarization using submodular optimization.

- **opt-network**: Refer to the README file in the `opt-network` directory for instructions on how to use the package for maximizing network contamination using submodular optimization.

Each README file provides specific usage instructions and code examples to help you quickly start using the respective packages.

Feel free to explore the PySubOpt project monorepo and leverage the submodular optimization packages for your own applications. If you have any questions or issues, please refer to the individual package's documentation or raise an issue in the repository.
