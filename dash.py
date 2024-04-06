import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import joblib
from sklearn.cluster import KMeans

st.set_page_config(page_title="Analisis Genre Film", page_icon=":film_projector:")

file_path_clf = 'gnb.pkl'
with open(file_path_clf, 'rb') as f:
    clf = joblib.load(f)

file_path_kmeans = 'kmeans.pkl'
with open(file_path_kmeans, 'rb') as f:
    kmeans = joblib.load(f)

url_data_clean_before_mapping = 'https://raw.githubusercontent.com/nurulvita/Data-Mining/main/Data%20Clean%20Before%20Mapping%20(1).csv'

genre_descriptions = {
    "Adventure": "Genre film yang menampilkan petualangan dan eksplorasi, seringkali di lokasi yang eksotis atau fantastis. Film-film dalam genre ini mengangkat tema-tema seperti penemuan diri, kepahlawanan, dan perjuangan melawan kekuatan jahat. Mereka sering mengikuti perjalanan sekelompok karakter yang berani dan berpetualang, menghadapi berbagai rintangan dan bahaya dalam pencarian mereka untuk mencapai tujuan akhir. Beberapa contoh film petualangan yang terkenal termasuk 'The Lord of the Rings' dan 'Indiana Jones'.",
    "Action": "Genre film yang menampilkan pertarungan, kejar-kejaran, dan aksi fisik yang dramatis. Film-film dalam genre ini sering menampilkan protagonis yang memiliki keterampilan bertarung yang luar biasa dan terlibat dalam konflik yang intens, seringkali dengan musuh atau penjahat yang kuat dan jahat. Mereka memacu adrenalin penonton dengan adegan-adegan aksi yang spektakuler dan efek khusus yang mengesankan. Beberapa contoh film aksi yang terkenal termasuk 'Die Hard' dan 'Mad Max: Fury Road'.",
    "Drama": "Genre film yang menampilkan konflik emosional dan karakter yang kompleks. Film-film dalam genre ini fokus pada pengembangan karakter dan interaksi antar karakter, seringkali menggali tema-tema seperti cinta, kehilangan, keberhasilan, dan kegagalan dalam kehidupan sehari-hari. Mereka menyoroti kehidupan manusia dengan cara yang mendalam dan menyentuh hati, memungkinkan penonton untuk merasakan emosi yang kuat dan mendalam. Beberapa contoh film drama yang terkenal termasuk 'The Shawshank Redemption' dan 'Forrest Gump'.",
    "Comedy": "Genre film yang bertujuan untuk menghibur penonton dengan humor dan kekonyolan. Film-film dalam genre ini sering menggunakan situasi lucu, dialog kocak, dan karakter-karakter konyol untuk menghasilkan tawa dan membuat penonton merasa bahagia. Mereka menyuguhkan keceriaan dan kegembiraan, menghadirkan momen-momen yang menyenangkan dan menghibur. Beberapa contoh film komedi yang terkenal termasuk 'The Hangover' dan 'Superbad'.",
    "Thriller or Suspense": "Genre film yang menampilkan ketegangan dan kecemasan yang tinggi. Film-film dalam genre ini sering menggabungkan unsur-unsur misteri, intrik, dan bahaya yang membuat penonton terpaku pada layar, tidak sabar untuk mengetahui apa yang akan terjadi selanjutnya. Mereka menciptakan suasana tegang dan tegang, menghadirkan plot yang rumit dan tak terduga. Beberapa contoh film thriller yang terkenal termasuk 'The Silence of the Lambs' dan 'Inception'.",
    "Horror": "Genre film yang bertujuan untuk menimbulkan ketakutan dan kecemasan pada penonton. Film-film dalam genre ini sering mengandalkan atmosfer gelap, musik mencekam, dan penggunaan efek khusus untuk menciptakan suasana yang menakutkan dan mencekam. Mereka memanfaatkan rasa takut manusia terhadap hal-hal yang tidak diketahui dan menghadirkan adegan-adegan yang menegangkan dan menyeramkan. Beberapa contoh film horor yang terkenal termasuk 'The Exorcist' dan 'The Shining'.",
    "Romantic Comedy": "Genre film yang menekankan pada hubungan romantis antara karakter utama dengan unsur humor. Film-film dalam genre ini sering menggabungkan kisah cinta yang manis dan lucu dengan momen-momen kocak dan situasi yang konyol, menghasilkan kombinasi yang menghibur dan menghangatkan hati. Mereka menyoroti dinamika hubungan percintaan dengan cara yang menggelitik dan menghibur, memungkinkan penonton untuk tertawa dan terinspirasi oleh cerita-cerita cinta yang unik. Beberapa contoh film rom-com yang terkenal termasuk 'When Harry Met Sally' dan 'Notting Hill'.",
    "Documentary": "Genre film yang memberikan dokumentasi atau rekaman factual tentang kejadian, kehidupan, atau fenomena. Film-film dalam genre ini sering bertujuan untuk memberikan wawasan yang mendalam tentang subjek tertentu, menggali fakta-fakta dan cerita-cerita yang menarik. Mereka menyajikan informasi yang akurat dan bermanfaat, memungkinkan penonton untuk memahami dunia di sekitar mereka dengan lebih baik. Beberapa contoh film dokumenter yang terkenal termasuk 'March of the Penguins' dan 'Bowling for Columbine'.",
    "Dark Comedy": "Genre film yang menggabungkan unsur komedi dengan tema gelap, kontroversial, atau tabu. Film-film dalam genre ini sering menggunakan humor yang gelap dan ironis untuk menghadirkan cerita-cerita yang menggugah pemikiran dan mengkritik keadaan sosial. Mereka mengeksplorasi tema-tema yang kompleks dan kontroversial dengan cara yang lucu namun tajam, memungkinkan penonton untuk merenungkan makna di balik kekonyolan. Beberapa contoh film dark comedy yang terkenal termasuk 'Dr. Strangelove' dan 'Fargo'.",
    "Musical": "Genre film yang menampilkan adegan bernyanyi dan menari sebagai bagian penting dari narasi. Film-film dalam genre ini sering menyajikan cerita-cerita yang diberi warna oleh lagu-lagu dan tarian-tarian yang menggugah semangat, seringkali dengan elemen dramatis atau romantis. Mereka memungkinkan karakter-karakter untuk menyampaikan emosi mereka melalui musik, menciptakan pengalaman sinematik yang memikat dan menggugah. Beberapa contoh film musikal yang terkenal termasuk 'The Sound of Music' dan 'La La Land'.",
    "Western": "Genre film yang berlatar belakang Amerika Serikat Barat pada abad ke-19, dengan cerita tentang koboi dan pertempuran hukum. Film-film dalam genre ini sering menampilkan aksi pistol, perburuan, dan konflik antara pahlawan dan penjahat dalam lanskap yang luas dan berdebu. Mereka mengangkat tema-tema seperti keadilan, petualangan, dan keberanian, membawa penonton ke dunia yang penuh dengan pahlawan legendaris dan pertempuran epik. Beberapa contoh film western yang terkenal termasuk 'The Good, the Bad and the Ugly' dan 'Django Unchained'.",
    "Concert or Performance": "Genre film yang merekam penampilan langsung dari konser musik, panggung, atau pertunjukan lainnya. Film-film dalam genre ini sering memberikan kesempatan kepada penonton untuk merasakan pengalaman konser secara langsung, menampilkan bakat-bakat musik dan hiburan dalam format yang berbeda. Mereka memungkinkan penonton untuk menyaksikan pertunjukan yang luar biasa dari kenyamanan rumah mereka sendiri, menangkap kegembiraan dan kegembiraan dari acara-acara langsung yang tak terlupakan. Beberapa contoh film konser yang terkenal termasuk 'Woodstock' dan 'Stop Making Sense'.",
    "Multiple Genres": "Genre film yang menggabungkan beberapa genre dalam satu film. Film-film dalam genre ini sering menggabungkan unsur-unsur cerita dari berbagai genre untuk menciptakan pengalaman yang unik dan beragam bagi penonton. Mereka dapat menyajikan campuran aksi, komedi, drama, dan elemen-elemen lain dalam satu narasi yang kompleks dan menarik. Beberapa contoh film multiple genres yang terkenal termasuk 'Pulp Fiction' dan 'The Matrix'.",
    "Reality": "Genre film yang menampilkan kehidupan nyata, seringkali dalam format dokumenter atau permainan realitas. Film-film dalam genre ini sering mengeksplorasi kehidupan sehari-hari orang-orang biasa atau menampilkan situasi-situasi yang diambil langsung dari kehidupan nyata. Mereka memberikan wawasan yang unik dan mendalam tentang manusia dan dunia di sekitar mereka, memungkinkan penonton untuk merasa terhubung dengan pengalaman-pengalaman yang autentik dan relevan. Beberapa contoh film reality yang terkenal termasuk 'Survivor' dan 'The Real World'."
}



st.title('Analisis Angka Profitabilitas dan Prediksi Keberhasilan Genre Film')

selected_page = st.sidebar.selectbox(
    "Select Page",
    ["Dashboard", "Data Distribution", "Category Analysis", "Relationship Analysis", "Composition Analysis", "Data Prediction", "Data Clustering"]
)

if selected_page == "Dashboard":
    data = pd.read_csv(url_data_clean_before_mapping)
    
    st.subheader("Genre Film")
    st.image('genrefilm.jpg', caption='Genre Film', use_column_width=True)
    
    st.subheader("Penjelasan Genre Film")
    selected_genre = st.selectbox('Select Genre', list(genre_descriptions.keys()))
    st.markdown(f"**{selected_genre}**: {genre_descriptions[selected_genre]}")

elif selected_page == "Data Distribution":
    data = pd.read_csv(url_data_clean_before_mapping)
    st.subheader("Data Distribution Section")

    feature_options = ['Movies Released', 'Gross', 'Tickets Sold']
    selected_feature = st.selectbox('Select Feature', feature_options)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(data[selected_feature], kde=True, color='skyblue', ax=ax)
    plt.title(f'Distribution of {selected_feature}')
    plt.xlabel(selected_feature)
    plt.ylabel('Frequency')
    st.pyplot(fig)

    st.write(f"Analisis ini menampilkan distribusi dari fitur {selected_feature} dalam dataset.")

elif selected_page == "Category Analysis":
    data = pd.read_csv(url_data_clean_before_mapping)
    st.subheader("Top Movie Gross vs Gross by Genre")
    fig = px.scatter(data, x='Gross', y='Tickets Sold', color='Genre', 
                     title='Hasil Pendapatan Kotor yang dihasilkan oleh genre film tiap tahun (1995 - 2018).',
                     labels={'Gross': 'Top Movie Gross', 'Tickets Sold': 'Gross', 'Genre': 'Genre'})
    fig.update_layout(xaxis_title='Top Movie Gross', yaxis_title='Tickets Sold')
    st.plotly_chart(fig)

    st.subheader("Histogram of Gross and Tickets Sold")
    plt.figure(figsize=(10, 6))
    sns.histplot(data['Gross'], bins=20, kde=True, color='skyblue', edgecolor='black')
    plt.title('Histogram of Gross')
    plt.xlabel('Gross')
    plt.ylabel('Frequency')
    st.pyplot(plt)

    plt.figure(figsize=(10, 6))
    sns.histplot(data['Tickets Sold'], bins=20, kde=True, color='salmon', edgecolor='black')
    plt.title('Histogram of Tickets Sold')
    plt.xlabel('Tickets Sold')
    plt.ylabel('Frequency')
    st.pyplot(plt)

elif selected_page == "Relationship Analysis":
    data = pd.read_csv(url_data_clean_before_mapping)
    st.subheader("Relationship Analysis Section")
    
    numeric_data = data.select_dtypes(include='number')
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(numeric_data.corr(), annot=True, cmap='viridis')
    plt.title('Heatmap of Feature Correlation')
    st.pyplot(plt.gcf())

    st.write("Visualisasi ini merupakan heatmap korelasi antara fitur-fitur numerik dalam dataset. Korelasi menggambarkan seberapa kuat hubungan linier antara dua variabel. Nilai korelasi berkisar dari -1 hingga 1, di mana nilai 1 menunjukkan korelasi positif sempurna, nilai -1 menunjukkan korelasi negatif sempurna, dan nilai 0 menunjukkan tidak adanya korelasi. Anotasi pada heatmap menunjukkan nilai korelasi antara dua fitur.")

    st.write("Berdasarkan heatmap korelasi, kita dapat melihat hubungan antara fitur-fitur numerik dalam dataset. Korelasi positif antara dua fitur menunjukkan bahwa mereka cenderung bergerak bersama-sama, sedangkan korelasi negatif menunjukkan bahwa mereka cenderung bergerak ke arah yang berlawanan. Dengan memahami korelasi antara fitur-fitur, kita dapat mendapatkan wawasan tentang bagaimana fitur-fitur tersebut berinteraksi satu sama lain dalam dataset.")

elif selected_page == "Composition Analysis":
    data = pd.read_csv(url_data_clean_before_mapping)
    st.subheader("Composition Analysis Section")

    composition_data = data['Year'].value_counts().reset_index()
    composition_data.columns = ['Year', 'Count']

    composition_data = composition_data.sort_values('Year')

    fig = px.bar(composition_data, x='Year', y='Count', 
                 title='Composition Analysis: Number of Movies Released by Year',
                 labels={'Year': 'Year', 'Count': 'Number of Movies Released'})
    st.plotly_chart(fig)

    st.write("Visualisasi ini menunjukkan komposisi jumlah film berdasarkan tahun rilisnya. Dengan melihat grafik ini, Anda dapat melihat distribusi jumlah film dalam rentang waktu yang dipilih. Hal ini memberikan pemahaman tentang perkembangan industri film dari waktu ke waktu, tren peningkatan atau penurunan produksi film, serta pola-pola yang mungkin teridentifikasi dari data.")

    st.write("Berdasarkan visualisasi ini, dapat dilihat bahwa jumlah film yang dirilis cenderung meningkat dari tahun ke tahun, dengan peningkatan yang signifikan terjadi pada beberapa periode tertentu. Hal ini menunjukkan bahwa industri film terus berkembang dan aktif dalam menghasilkan konten baru.")

elif selected_page == "Data Prediction":
    data = pd.read_csv(url_data_clean_before_mapping)
    st.subheader("Predicting Ticket Sales")

    genre_options = data['Genre'].unique()
    selected_genre = st.selectbox('Genre', genre_options)

    year_options = data['Year'].unique()
    selected_year = st.selectbox('Year', year_options)

    min_movies_released = int(data['Movies Released'].min())
    max_movies_released = int(data['Movies Released'].max())
    selected_movies_released = st.slider('Movies Released', min_movies_released, max_movies_released)

    min_gross = int(data['Gross'].min())
    max_gross = int(data['Gross'].max())
    selected_gross = st.slider('Gross', min_gross, max_gross)

    min_tickets_sold = int(data['Tickets Sold'].min())
    max_tickets_sold = int(data['Tickets Sold'].max())
    selected_tickets_sold = st.slider('Tickets Sold', min_tickets_sold, max_tickets_sold)
    
    tickets_sold_category_options = ['Low', 'Medium', 'High']
    selected_tickets_sold_category = st.selectbox('Tickets Sold Category', tickets_sold_category_options)

    tickets_sold_category_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
    selected_tickets_sold_category = tickets_sold_category_mapping[selected_tickets_sold_category]

    passenger = pd.DataFrame({
        'Genre': [selected_genre],
        'Year': [selected_year],
        'Movies Released': [selected_movies_released],
        'Gross': [selected_gross],
        'Tickets Sold': [selected_tickets_sold],
        'Tickets Sold Category': [selected_tickets_sold_category]
    })

    passenger['Genre'] = passenger['Genre'].astype('category').cat.codes
    

    st.subheader("Kesimpulan Data Prediktif:")
    prediction_state = st.markdown('Predicting...')
    predicted_tickets_sold = clf.predict(passenger)
    predicted_tickets_sold_formatted = '{:,.0f}'.format(predicted_tickets_sold[0])
    prediction_state.markdown(f"Prediksi kategori penjualan tiket: {predicted_tickets_sold_formatted}. Prediksi ini didasarkan pada masukan pengguna dan model yang dilatih.")
    
elif selected_page == "Data Clustering":
    st.subheader("Performing Data Clustering")
    
    num_clusters = st.slider("Number of Clusters", 2, 10, 3)

    data = pd.read_csv(url_data_clean_before_mapping)

    X = data[['Gross', 'Tickets Sold']]
    kmeans = KMeans(n_clusters=num_clusters)
    clusters = kmeans.fit_predict(X)

    data['Cluster'] = clusters

    fig = px.scatter(data, x='Gross', y='Tickets Sold', color='Cluster', 
                     title='Data Clustering: Gross vs Tickets Sold')
    st.plotly_chart(fig)

    st.write("Visualisasi ini menunjukkan hasil dari analisis data clustering berdasarkan variabel Gross dan Tickets Sold. Data telah dikelompokkan ke dalam beberapa klaster berdasarkan karakteristik yang serupa dalam hal pendapatan kotor (Gross) dan jumlah tiket yang terjual (Tickets Sold)")

st.sidebar.markdown("---")
st.sidebar.markdown("Created by Nurul Vita Azizah")
