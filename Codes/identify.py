import face_recognition
from PIL import Image, ImageDraw

image_of_arteta = face_recognition.load_image_file(r'C:\Users\user\Desktop\Online Learning\Image Face Recognition\Arsenal Players\Arteta.jpg')
arteta_face_encoding = face_recognition.face_encodings(image_of_arteta)[0]

image_of_martinelli = face_recognition.load_image_file(r'C:\Users\user\Desktop\Online Learning\Image Face Recognition\Arsenal Players\Martinelli.png')
martinelli_face_encoding = face_recognition.face_encodings(image_of_martinelli)[0]

image_of_bellerin = face_recognition.load_image_file(r'C:\Users\user\Desktop\Online Learning\Image Face Recognition\Arsenal Players\Bellerin.jpg')
bellerin_face_encoding = face_recognition.face_encodings(image_of_bellerin)[0]

image_of_ceballos = face_recognition.load_image_file(r'C:\Users\user\Desktop\Online Learning\Image Face Recognition\Arsenal Players\Ceballos.jpg')
ceballos_face_encoding = face_recognition.face_encodings(image_of_ceballos)[0]

image_of_saka = face_recognition.load_image_file(r'C:\Users\user\Desktop\Online Learning\Image Face Recognition\Arsenal Players\Saka.png')
saka_face_encoding = face_recognition.face_encodings(image_of_saka)[0]

image_of_partey = face_recognition.load_image_file(r'C:\Users\user\Desktop\Online Learning\Image Face Recognition\Arsenal Players\Partey.jpg')
partey_face_encoding = face_recognition.face_encodings(image_of_partey)[0]

image_of_tierny = face_recognition.load_image_file(r'C:\Users\user\Desktop\Online Learning\Image Face Recognition\Arsenal Players\Tierny.jpg')
tierny_face_encoding = face_recognition.face_encodings(image_of_tierny)[0]

#  Create arrays of encodings and names
known_face_encodings = [
  arteta_face_encoding,
  martinelli_face_encoding,
  bellerin_face_encoding,
  ceballos_face_encoding,
  saka_face_encoding,
  partey_face_encoding,
  tierny_face_encoding
  ]

known_face_names = [
  "Arteta",
  "Martinelli",
  "Bellerin",
  "Ceballos",
  "Saka",
  "Partey",
  "Tierny"
]

# Load test image to find faces in
# test_image = face_recognition.load_image_file(r'C:\Users\user\Desktop\Online Learning\Face Recognition\Testing_Data\EmqDDvOXIAIBymi.jpg')
test_image = face_recognition.load_image_file(r'C:\Users\user\Desktop\Online Learning\Image Face Recognition\Testing_Data\EiTq_3gWAAIH2RB.jpg')

# Find faces in test image
face_locations = face_recognition.face_locations(test_image)
face_encodings = face_recognition.face_encodings(test_image, face_locations)

# Convert to PIL format
pil_image = Image.fromarray(test_image)

# Create a ImageDraw instance
draw = ImageDraw.Draw(pil_image)

# Loop through faces in test image
for(top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
  matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

  name = "Unknown Person"

  # If match
  if True in matches:
    first_match_index = matches.index(True)
    name = known_face_names[first_match_index]

  # Draw box
  draw.rectangle(((left, top), (right, bottom)), outline=(255,255,0))

  # Draw label
  text_width, text_height = draw.textsize(name)
  draw.rectangle(((left,bottom - text_height - 10), (right, bottom)), fill=(255,255,0), outline=(255,255,0))
  draw.text((left + 6, bottom - text_height - 5), name, fill=(0,0,0))

del draw

# Display image
pil_image.show()

# Save image
# pil_image.save('identify.jpg')
