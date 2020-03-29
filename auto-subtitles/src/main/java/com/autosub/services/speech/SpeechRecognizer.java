package com.autosub.services.speech;

import com.google.cloud.speech.v1.SpeechRecognitionResult;
import com.google.protobuf.ByteString;
import org.jetbrains.annotations.NotNull;

import java.io.IOException;
import java.util.List;

public interface SpeechRecognizer {
    @NotNull
    List<SpeechRecognitionResult> recognize(@NotNull final ByteString content) throws IOException;
}
