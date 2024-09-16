class Constants:
    # Question Prefix.
    questionPrefix = "Q : "

    # Answer Prefix.
    answerPrefix = "A : "

    # Section Renaming Regex
    sectionRenameRegex = r'^(\d+\.)+\d*\s*'

    ###################### Parser Constants ######################

    # Section Start Keywords
    sectionStartKeywords = ["method", "methodology", "our work", "approach"]

    # Section End Keywords
    sectionEndKeywords = ["results", "conclusion", "ablations", "training", "evaluation", "fine tuning"]

    # Section Before Start Keywords
    sectionBeforeStartKeywords = ["introduction", "abstract", "existing work", "previous work", "related work", "datasets", "introduction and related work"]

    # Default Section for outlier papers.
    defaultSection = "abstract"

    # default text for section in case of errors.
    defaultSectionText = "No text available for this section from the paper."
