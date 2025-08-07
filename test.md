# Markdown Test Case

Welcome to the **Markdown Test Case**. This document is designed to test various Markdown elements in a verbose, structured format. It includes:

- Headers
- Paragraphs
- Emphasis (bold, italic)
- Blockquotes
- Lists (ordered and unordered)
- Code blocks
- Tables
- Links
- Images
- Horizontal rules

---

## 1. Headers

### 1.1 Subheaders

This section tests header rendering. Each level should appear progressively smaller:

#### Level 4 Header
##### Level 5 Header
###### Level 6 Header

---

## 2. Paragraphs and Emphasis

This is a paragraph with _italic text_, **bold text**, and **_bold italic text_**.

Here is some `inline code` inside a sentence to show syntax highlighting or lack thereof.

---

## 3. Blockquotes

> This is a blockquote.
> 
> Blockquotes can span multiple lines and contain nested elements.
>
> > Nested blockquote example.

---

## 4. Lists

### 4.1 Unordered List

- Item 1
  - Subitem 1.1
    - Sub-subitem 1.1.1
- Item 2

### 4.2 Ordered List

1. First item
2. Second item
   1. Subitem 2.1
   2. Subitem 2.2
3. Third item

---

## 5. Code Blocks

Here is a multi-line code block in **Python**:

```python
import math

def calculate_circle_area(radius):
    """Returns the area of a circle given its radius."""
    if radius <= 0:
        return 0
    return math.pi * radius ** 2

print(calculate_circle_area(5))
```