package com.autosub.utils;

import com.google.cloud.speech.v1.SpeechRecognitionAlternative;
import com.google.cloud.speech.v1.WordInfo;
import com.google.protobuf.util.Durations;
import org.jetbrains.annotations.NotNull;

import java.util.List;
import java.util.stream.Collectors;

public final class SpeechRecognitionUtils {

    private SpeechRecognitionUtils() {
        // Not instantiable
    }

    public static long startTimeOf(@NotNull final WordInfo wordInfo) {
        return Durations.toMillis(wordInfo.getStartTime());
    }

    public static long startTimeOf(@NotNull final SpeechRecognitionAlternative alt) {
        return startTimeOf(alt.getWords(0));
    }

    public static long endTimeOf(@NotNull final WordInfo wordInfo) {
        return Durations.toMillis(wordInfo.getEndTime());
    }

    public static long endTimeOf(@NotNull final SpeechRecognitionAlternative alt) {
        return endTimeOf(alt.getWords(alt.getWordsCount() - 1));
    }

    @NotNull
    public static String subtitleFrom(@NotNull final List<WordInfo> words) {
        return words.stream()
                .map(WordInfo::getWord)
                .collect(Collectors.joining(" "));
    }
}
