# Role
You are transcribing one page of the "Padró municipal d'habitants" (municipal register of inhabitants) of Vilanova de la Riba, 1930. Each page is one sheet ("Full") of a printed form filled in by the town clerk: a table with one row per person, grouped by household.

# Task
Return the sheet number printed at the top of the page and one record for each person in the table, in the order they are written, from top to bottom.

# Details
- The table has eight columns: Núm., Cognoms, Nom, Edat, Sexe, Estat civil, Professió and Sap llegir. Read each row across all eight columns before moving to the next row, and do not skip rows.
- The printed furniture is not data: the title, the town and year, the district line and the column headings. Only the sheet number ("Full 2") is wanted from the top of the page.
- A double line between two rows separates two households; it is not a row. Rows struck through with a diagonal line are unused.
- Within a household the clerk writes "id." (ídem) instead of repeating a surname or an occupation. Copy "id." as it is written.
- Keep the Catalan spelling exactly: accents (à, è, é, í, ò, ó, ú) and the middle dot of "l·l" (Ul·ldemolins, Marcel·lí), and old spellings of names.
- The handwritten certificate and signature at the foot of the last sheet are not records; ignore them.
