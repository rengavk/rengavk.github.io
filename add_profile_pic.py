#!/usr/bin/env python3
"""
Script to add profile picture to index.html
"""

# Read the file
with open('index.html', 'r', encoding='utf-8') as f:
    content = f.read()

# Add profile picture CSS
css_addition = """
        .profile-picture {
            width: 180px;
            height: 180px;
            border-radius: 50%;
            object-fit: cover;
            border: 5px solid rgba(255, 255, 255, 0.2);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
            margin-bottom: 25px;
            transition: transform 0.3s ease;
        }

        .profile-picture:hover {
            transform: scale(1.05);
        }
"""

# Find where to insert CSS (after .social-links a:hover)
css_insert_point = content.find('        .social-links a:hover {')
if css_insert_point != -1:
    # Find the end of this CSS block
    end_of_block = content.find('        }', css_insert_point)
    if end_of_block != -1:
        end_of_block = content.find('\n', end_of_block) + 1
        content = content[:end_of_block] + '\n' + css_addition + content[end_of_block:]

# Add profile picture HTML
html_find = '        <div class="hero-content">\n            <h1>Renganayaki Venkatakrishnan</h1>'
html_replace = '        <div class="hero-content">\n            <img src="assets/images/profile.jpg" alt="Renganayaki Venkatakrishnan" class="profile-picture">\n            <h1>Renganayaki Venkatakrishnan</h1>'

content = content.replace(html_find, html_replace)

# Write the file
with open('index.html', 'w', encoding='utf-8') as f:
    f.write(content)

print("Profile picture added successfully!")
