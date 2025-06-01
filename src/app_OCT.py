import torch
from torchvision import transforms
from transformers import ViTForImageClassification, AutoConfig, ViTImageProcessor
from PIL import Image
from openai import OpenAI
import pandas as pd
import streamlit as st


class CustomViTForImageClassification(ViTForImageClassification):
    def __init__(self, config):
        super().__init__(config)
        self.classifier = torch.nn.Linear(config.hidden_size, 8)


# Загрузка модели и feature extractor
model_name = "src/vit_model/config.json/pytorch_model.bin"
config = AutoConfig.from_pretrained('src/vit_model/config.json')

config.num_labels = 8
model = CustomViTForImageClassification(config)
feature_extractor = ViTImageProcessor.from_pretrained(
    'src/vit_model/preprocessor_config.json')

# Загрузка сохраненных весов
model.load_state_dict(torch.load('src/vit_model/config.json/vitalik.pth',
                                 map_location=torch.device('cpu')))
model.eval()

df = pd.read_excel('/home/olga/Рабочий стол/Вкр/pythonProject/тест.xlsx')


def deepseek_pred(promt):
    client = OpenAI(api_key="", base_url="https://api.deepseek.com") # добавить api_key

    response = client.chat.completions.create(
        model="deepseek-chat",  # определение модели
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},  
            {"role": "user", "content": promt},  
        ],
        stream=False  
    )

    return (response.choices[0].message.content)


def picture_prepare(img_source):
    img = Image.open(img_source).convert("RGB")

    data_transforms = transforms.Compose([
        transforms.Resize(180),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = data_transforms(img).unsqueeze(0)
    inputs = {'pixel_values': input_tensor}
    return inputs


def vitalik_pred(inputs):
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_labels = torch.sigmoid(logits).numpy().flatten()  # Для multi-label классификации
        print(f"Predicted labels: {predicted_labels}")

    label_names = ['IRF', 'SRF', 'HE', 'CNV', 'PED', 'Drusen', 'ERM', 'MH']
    biomarcers = []
    for i in range(len(predicted_labels)):
        if predicted_labels[i] > 0.35:
            biomarcers.append(label_names[i])

    return biomarcers


def load_image(uploaded_file):
    image = Image.open(uploaded_file)
    return image


# основное тело программы
# Настройка боковой панели
st.sidebar.image(Image.open("logo.jpg"), width=500)
st.sidebar.title("Добро пожаловать в OCTassistant!")

st.sidebar.info("OCTassistant поможет вам в дифференциальной диагностике заболеваний глаз. ")
uploaded_img = st.sidebar.file_uploader("Загрузите изображение OCT макулярной зоны сетчатки", type=["jpg"])

# главное окно

st.header("OCTassistant")
st.markdown(
    "Чтобы получить 3 дифференциальных диагноза, загрузите изображение OCT  и заполните поля слева."
)
if uploaded_img:
    img = load_image(uploaded_img)
    st.image(img, width=500, caption="Ваше изображение")
patient_complaints = st.sidebar.text_area('Напишите  пол, возраст и  основные жалобы пациента в свободной форме.',
                                          height=68)
diabet = st.sidebar.selectbox('Есть ли у пациента диабет?', ['да', 'нет'])
other_diagnoses = st.sidebar.text_area("Напишите другие важные данные анамнеза пациента, например, перенесенные "
                                       "операции, "
                                       "хронические заболевания, принимаемые препараты.", height=68)
if st.button("Получить 3 вероятных диагноза"):
    img_prep = picture_prepare(uploaded_img)  # Теперь передаем uploaded_img напрямую
    img_predict = vitalik_pred(img_prep)
    biomarkers_str = ', '.join(img_predict) if img_predict else "биомаркеры не обнаружены"
    promt = (patient_complaints + ' диабет-' + diabet + other_diagnoses + ' на снимке OCT модулярной зоны выявлено: '
             + biomarkers_str + ' Напиши кратко 3 наиболее вероятных диагноза.')
    tree_diag = deepseek_pred(promt)
    st.success(f'На OCT выявлены биомаркеры: {biomarkers_str},\n {tree_diag}')
