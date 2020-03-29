package com.autosub.services.speech.google;

import com.autosub.services.speech.SpeechRecognizer;
import com.google.cloud.speech.v1.RecognitionAudio;
import com.google.cloud.speech.v1.SpeechClient;
import com.google.cloud.speech.v1.SpeechRecognitionResult;
import com.google.protobuf.ByteString;
import org.jetbrains.annotations.NotNull;

import java.io.IOException;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ExecutionException;

public class GoogleS2T implements SpeechRecognizer {
    @NotNull private GoogleS2TConfig googleConfig;

    private GoogleS2T(@NotNull final GoogleS2TConfig config) {
        this.googleConfig = config;
    }

    @NotNull
    public static GoogleS2T of(@NotNull final GoogleS2TConfig config) {
        return new GoogleS2T(config);
    }

    @NotNull
    @Override
    public List<SpeechRecognitionResult> recognize(@NotNull final ByteString content) throws IOException {
        final var config = googleConfig.getConfig();
        final var audio = RecognitionAudio.newBuilder()
                .setContent(content)
                .build();
        try (final var speech = SpeechClient.create()) {
            final var response = speech.longRunningRecognizeAsync(config, audio);
            while (!response.isDone()) {
                Thread.sleep(10000);
            }
            return response.get().getResultsList();
        } catch (InterruptedException | ExecutionException e) {
            return Collections.emptyList();
        }
    }
}
