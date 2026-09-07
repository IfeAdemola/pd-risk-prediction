from pd_risk.modelling.reporting import (
    ExperimentReporter,
)

from pd_risk.modelling.comparison import (
    ExperimentComparator
)
from pathlib import Path

def main():

    # Use the same comparator construction
    # you already use in test_comparison.py.

    comparator = ExperimentComparator(
            "output/experiments"
        )

    comparison = (
        comparator
        .build_comparison()
    )

    reporter = ExperimentReporter(
        comparator
    )

    report = reporter.build_report(
        comparison
    )

    markdown = (
        reporter.to_markdown(
            report
        )
    )

    print(
        "\nMarkdown report:"
    )

    print(
        markdown
    )


    # Save the report to a file.
    output_path = (
        Path(
            "output"
        )
        / "comparison"
        / "comparison_report.md"
    )

    saved_path = (
        reporter.save_markdown(
            report,
            output_path,
        )
    )

    print(
        "\nReport saved to:"
    )

    print(
        saved_path
    )

if __name__ == "__main__":
    main()