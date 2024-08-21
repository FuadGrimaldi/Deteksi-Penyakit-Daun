num_classes = [
    'Daun_Apel_Apel_scab', 'Daun_Apel_Busuk_hitam', 'Daun_Apel_Karangan_apel', 'Daun_Apel_sehat', 'Daun_Blueberry_sehat', 
    'Daun_Ceri_(termasuk_asam)_Jamur_pudarnya', 'Daun_Ceri_(termasuk_asam)_sehat', 
    'Daun_Jagung_Bercak_daun_Cercospora_Bercak_daun_abu', 'Daun_Jagung_Jamur_umum', 
    'Daun_Jagung_Bercak_daun_ular', 'Daun_Jagung_sehat', 'Daun_Anggur_Busuk_hitam', 
    'Daun_Anggur_Esca_(Jamur_Hitam)', 'Daun_Anggur_Bercak_daun_(Isariopsis_Leaf_Spot)', 'Daun_Anggur_sehat', 
    'Daun_Jeruk_Haunglongbing_(Penyakit_greening_Citrus)', 'Daun_Persik_Bercak_bakteri', 'Daun_Persik_sehat',
    'Daun_Paprika_bell_Bercak_bakteri', 'Daun_Paprika_bell_sehat', 'Daun_Kentang_Bercak_awal', 
    'Daun_Kentang_Bercak_akhir', 'Daun_Kentang_sehat', 'Daun_Raspberry_sehat', 'Daun_Kedelai_sehat', 
    'Daun_Labuh_Jamur_pudarnya', 'Daun_Stroberi_Bercak_daun', 'Daun_Stroberi_sehat', 'Daun_Tomat_Bercak_bakteri', 
    'Daun_Tomat_Bercak_awal', 'Daun_Tomat_Bercak_akhir', 'Daun_Tomat_Jamur_daun', 'Daun_Tomat_Bercak_Septoria', 
    'Daun_Tomat_Kutu_spider_Dua_titik', 'Daun_Tomat_Bercak_Target', 
    'Daun_Tomat_Virus_Kerut_Daun_Kuning_Tomat', 'Daun_Tomat_Virus_mosaik_Tomat', 'Daun_Tomat_sehat'
]

disease_dic = {
    'Daun_Apel_Apel_scab': "Jamur *Venturia inaequalis* menyebabkan penyakit apple scab, yang mengakibatkan lesi gelap dan cekung pada daun, buah, dan batang.",
    
    'Daun_Apel_Busuk_hitam': "Busuk hitam pada apel disebabkan oleh jamur *Diplodia seriata*, yang menyebabkan lesi gelap dan cekung yang akhirnya menyebabkan pembusukan buah.",
    
    'Daun_Apel_Karangan_apel': "Kanker apel disebabkan oleh jamur *Neonectria ditissima*, yang menyebabkan lesi cekung dan kasar pada batang dan cabang.",
    
    'Daun_Apel_sehat': "Daun apel yang sehat tidak menunjukkan tanda-tanda penyakit dan bebas dari lesi atau perubahan warna.",
    
    'Daun_Blueberry_sehat': "Daun blueberry yang sehat berwarna cerah dan bebas dari gejala penyakit seperti bercak atau perubahan warna.",
    
    'Daun_Ceri_(termasuk_asam)_Jamur_pudarnya': "Jamur bubuk pada ceri, disebabkan oleh *Podosphaera clandestina*, menghasilkan pertumbuhan jamur putih seperti bubuk pada daun dan tunas.",
    
    'Daun_Ceri_(termasuk_asam)_sehat': "Daun ceri yang sehat bebas dari gejala penyakit dan terlihat hijau dan segar.",
    
    'Daun_Jagung_Bercak_daun_Cercospora_Bercak_daun_abu': "Bercak daun abu pada jagung, disebabkan oleh *Cercospora zeae-maydis*, mengakibatkan lesi abu-abu dengan halo kuning pada daun.",
    
    'Daun_Jagung_Jamur_umum': "Penyakit jamur umum pada jagung termasuk karat dan jamur, yang menyebabkan berbagai gejala seperti perubahan warna dan deformasi.",
    
    'Daun_Jagung_Bercak_daun_ular': "Bercak daun jagung akibat *Helminthosporium turcicum* menghasilkan lesi gelap memanjang pada daun jagung.",
    
    'Daun_Jagung_sehat': "Daun jagung yang sehat menunjukkan warna hijau segar dan bebas dari gejala penyakit.",
    
    'Daun_Anggur_Busuk_hitam': "Busuk hitam pada anggur disebabkan oleh jamur *Guignardia bidwellii*, yang menyebabkan lesi hitam pada buah dan daun.",
    
    'Daun_Anggur_Esca_(Jamur_Hitam)': "Esca pada anggur disebabkan oleh jamur *Phaeomoniella chlamydospora* dan *Phaeoacremonium aleophilum*, yang menyebabkan pembusukan dan kematian jaringan.",
    
    'Daun_Anggur_Bercak_daun_(Isariopsis_Leaf_Spot)': "Bercak daun anggur disebabkan oleh jamur *Isariopsis griseola*, menghasilkan bercak coklat pada daun.",
    
    'Daun_Anggur_sehat': "Daun anggur yang sehat bebas dari lesi atau gejala penyakit dan terlihat hijau dan segar.",
    
    'Daun_Jeruk_Haunglongbing_(Penyakit_greening_Citrus)': "Penyakit greening pada jeruk disebabkan oleh bakteri *Candidatus Liberibacter asiaticus*, menyebabkan daun menguning dan buah kecil dan cacat.",
    
    'Daun_Persik_Bercak_bakteri': "Bercak bakteri pada persik disebabkan oleh *Xanthomonas campestris*, yang menyebabkan lesi basah pada daun dan buah.",
    
    'Daun_Persik_sehat': "Daun persik yang sehat bebas dari gejala penyakit dan tampak segar dan hijau.",
    
    'Daun_Paprika_bell_Bercak_bakteri': "Bercak bakteri pada paprika bell disebabkan oleh *Xanthomonas campestris*, yang menyebabkan bercak basah pada daun.",
    
    'Daun_Paprika_bell_sehat': "Daun paprika bell yang sehat bebas dari lesi atau gejala penyakit dan tampak hijau dan segar.",
    
    'Daun_Kentang_Bercak_awal': "Bercak awal pada kentang disebabkan oleh jamur *Alternaria solani*, yang menyebabkan bercak gelap dengan tepi kuning pada daun.",
    
    'Daun_Kentang_Bercak_akhir': "Bercak akhir pada kentang adalah stadium lanjut dari penyakit yang disebabkan oleh jamur *Alternaria solani*, yang menyebabkan lesi gelap besar pada daun.",
    
    'Daun_Kentang_sehat': "Daun kentang yang sehat bebas dari gejala penyakit dan tampak hijau dan segar.",
    
    'Daun_Raspberry_sehat': "Daun raspberry yang sehat menunjukkan warna hijau cerah dan bebas dari lesi atau gejala penyakit.",
    
    'Daun_Kedelai_sehat': "Daun kedelai yang sehat bebas dari bercak atau gejala penyakit dan terlihat hijau dan segar.",
    
    'Daun_Labuh_Jamur_pudarnya': "Jamur pudarnya pada labuh disebabkan oleh *Colletotrichum orbiculare*, yang menyebabkan bercak hitam dan kerusakan pada daun.",
    
    'Daun_Stroberi_Bercak_daun': "Bercak daun pada stroberi disebabkan oleh *Mycosphaerella fragariae*, yang menyebabkan bercak coklat dengan tepi kuning pada daun.",
    
    'Daun_Stroberi_sehat': "Daun stroberi yang sehat bebas dari gejala penyakit dan tampak hijau dan segar.",
    
    'Daun_Tomat_Bercak_bakteri': "Bercak bakteri pada tomat disebabkan oleh *Xanthomonas campestris*, yang menyebabkan bercak kecil dan lesi basah pada daun.",
    
    'Daun_Tomat_Bercak_awal': "Bercak awal pada tomat disebabkan oleh *Alternaria solani*, yang menyebabkan bercak gelap dengan tepi kuning pada daun.",
    
    'Daun_Tomat_Bercak_akhir': "Bercak akhir pada tomat adalah stadium lanjut dari penyakit yang disebabkan oleh *Alternaria solani*, yang menyebabkan lesi besar pada daun.",
    
    'Daun_Tomat_Jamur_daun': "Jamur daun pada tomat disebabkan oleh *Cladosporium fulvum*, yang menyebabkan bercak hitam pada daun.",
    
    'Daun_Tomat_Bercak_Septoria': "Bercak Septoria pada tomat disebabkan oleh *Septoria lycopersici*, menghasilkan bercak hitam dengan tepi kuning pada daun.",
    
    'Daun_Tomat_Kutu_spider_Dua_titik': "Kutu laba-laba dua titik pada tomat disebabkan oleh *Tetranychus urticae*, menyebabkan noda kuning dan kerusakan daun.",
    
    'Daun_Tomat_Bercak_Target': "Bercak target pada tomat disebabkan oleh *Alternaria solani*, yang menyebabkan bercak hitam dengan pola konsentris pada daun.",
    
    'Daun_Tomat_Virus_Kerut_Daun_Kuning_Tomat': "Virus kerut daun kuning tomat disebabkan oleh virus *Tomato yellow leaf curl virus* (TYLCV), menyebabkan daun mengerut dan menguning.",
    
    'Daun_Tomat_Virus_mosaik_Tomat': "Virus mosaik tomat disebabkan oleh *Tomato mosaic virus* (ToMV), yang menyebabkan pola mosaik dan perubahan warna pada daun.",
    
    'Daun_Tomat_sehat': "Daun tomat yang sehat bebas dari gejala penyakit dan tampak hijau dan segar."
}
