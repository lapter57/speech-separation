# speech-separation

## Overview
Этот проект нацелен на изучение области разделения речи, а именно выделение одного речего сигнала из смеси звуков. На данный момент данную проблему пытаются решить при помощи глубокого обучения (_DNN_). Одно из акутальных и интересных решений предложила команда исследователей из компании _Google_[1]. В данной работе они описывают две модели: 
* _Audio-only model_ - модель, которая обучается только при помощи аудио данных
* _Audio-Visual model_ - модель, которая принимает на вход вместе с аудио данными и видео данные, а именно обнаруженные лица, чтобы впоследствии разделить аудио на отдельные аудиопотоки для каждого обнаруженного говорящего.

В этом проекте попытаемся реализовать обе модели.

## Data

Для обучения моделей _Google_ разработал собственный _dataset_ - [_AVSpeech_](https://looking-to-listen.github.io/avspeech/index.html), который содержит 4700 часов видеоклипов по 3-10 с каждый с видимыми лицами говорящих и с их чистой речью. Для генерации фонового шума они предлагают использовать данные из известного _dataset_ - [_AudioSet_](https://research.google.com/audioset/index.html). Для обучения моделей требуется скачать 3 _csv_, которые предлагается поместить в папку `data/csv`:

* [_avpeech_train.csv_](https://storage.cloud.google.com/avspeech-files/avspeech_train.csv)
* [_avpeech_test.csv_](https://storage.cloud.google.com/avspeech-files/avspeech_test.csv)
* [_eval_segments.csv_](http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/eval_segments.csv) (для фонового шума)

Для того чтобы скачать видео данные или только их аудио часть, был написан скрипт `scripts/audio_downloader.py`. Выполните следующую команду, чтобы понять как его использовать: 

```bash
$ python3 scripts/audio_downloader.py --help
```

### Data for Audio-only model

Для обучения _Audio-only model_ требуются данные, которые можно сгенерировать при помощи скрипта `scripts/audio_data_builder.py`. Выполните следующую команду, чтобы понять как его использовать: 

```bash
$ python3 scripts/audio_data_builder.py --help
```
Данный скрипт сгененрирует три папки в указанной директории:  _clean_, _mix_, _crm_. Во всех папках сохранены массивы _numpy_ в формате `.npy`. 

* В папке _clean_ сохранена чистая речь говорящих. Формат файлов:
    - `idx:youtube_id.npy`
* В папке _mix_ сохранена смесь голосов людей. формат файлов:
    - без фонового шума (2 голоса): `idx1:youtube_id1.idx2:youtube_id2.npy` 
    - с фоновым шумом (2 голоса + шум): `idx1:youtube_id1.idx2:youtube_id2.n:idx3:youtube_id3.npy`
* В папке _crm_ хранятся данные о масках _cRM_ [2]. Формат файлов:
    - маска, которую если применить к смеси _mix:idx1:youtube_id1:idx2:youtube_id2_, то можно получить чистую речь _clean:idx1:youtube_id1_: `clean:idx1:youtube_id1 mix:idx1:youtube_id1:idx2:youtube_id2.npy`

## Audio-only model
Для обучения данной модели были сгенерированы аудиоданные (смеси голосов) длительностью 3 с, частотой дискретизации 16 кГц, стерео звук был преобразован в моно. Каждый аудио фрагмент преобразуется при помощи степенного сжатия (_power_law_), чтобы громкий звук не подавлял слабый. Затем к каждому 3-ех секундному фрагменту был применен алгоритм _STFT_ со использованием окна Ганна со следующими параметрами: window length = 25 ms, hop length = 10 ms, FFT size = 512. Поскольку в алгоритме _STFT_ мы работаем с полярными координатами, то размер входных аудио признаков для нашей модели в итоге равен _257\*258\*2_. Эти данные подаются на вход _CNN_, которая на выходе выдает _cRM_ для каждой обнаруженной речи. Применив последовательно каждую полученную _cRM_ к изначальному аудио, мы получим фрагменты чистой речи.


## Reference

[1] [Lookng to Listen at the Cocktail Party:A Speaker-Independent Audio-Visual Model for Speech Separation, A. Ephrat et al., arXiv:1804.03619v2 [cs.SD] 9 Aug 2018](https://github.com/Lapter57/speech-separation/blob/master/articles/Ephrat2018.pdf)

[2] [D.S. Williamson, Y. Wang, and D.L. Wang, "Complex ratio
masking for monaural speech separation," IEEE/ACM Trans.
Audio Speech Lang. Proc., vol. 24, pp. 483–492, 2016. ](https://github.com/Lapter57/speech-separation/blob/master/articles/Williamson2015.pdf)
