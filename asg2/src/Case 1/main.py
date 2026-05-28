"""Main entry point for Case 1.

Runs every task in order:
    Task 3.1 - Newton's method finds the analytic center of A x <= b.
    Task 3.2 - scaling the 5th constraint leaves the center unchanged.
    Task 3.3 - normalizing the constraints leaves the center unchanged.
    Task 4   - gradient descent vs. Newton on a larger barrier problem.
"""

import matplotlib.pyplot as plt

import task31
import task32
import task33
import task4


def main():
    """Run Tasks 3.1, 3.2, 3.3 and 4 in order and show the figures."""
    print("=" * 60)
    print("Task 3.1")
    print("=" * 60)
    x_star = task31.run()

    print()
    print("=" * 60)
    print("Task 3.2")
    print("=" * 60)
    task32.run(x_star)

    print()
    print("=" * 60)
    print("Task 3.3")
    print("=" * 60)
    task33.run(x_star)

    print()
    print("=" * 60)
    print("Task 4")
    print("=" * 60)
    task4.run()

    plt.show()


if __name__ == "__main__":
    main()
