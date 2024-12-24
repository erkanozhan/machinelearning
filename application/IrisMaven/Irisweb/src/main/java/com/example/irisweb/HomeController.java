package com.example.irisweb;

import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;

// HomeController Ayrı Bir Dosyada
@Controller
public class HomeController {

    @GetMapping("/")
    public String home() {
        return "predict"; // `src/main/resources/templates/predict.html` olmalı.
    }
}
