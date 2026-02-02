def course_advisor(course_title, duration_minutes, platform, views_per_day, engagement_rate, success_pred):
    suggestions = []

    if "beginner" not in course_title.lower() and "advanced" not in course_title.lower():
        suggestions.append("Add difficulty level in the title (Beginner / Intermediate / Advanced).")

    if "project" not in course_title.lower():
        suggestions.append("Include hands-on or project-based learning in the title.")

    if duration_minutes > 300:
        suggestions.append("Split the course into smaller modules to reduce learner fatigue.")

    if engagement_rate < 0.02:
        suggestions.append("Add quizzes, assignments, and interactive sessions to improve engagement.")

    if success_pred == 0:
        suggestions.append("Improve thumbnail, title SEO, and add real-world case studies.")

    if platform.lower() == "youtube":
        suggestions.append("Optimize YouTube SEO and use attractive thumbnails.")

    return suggestions

