var Crawler = require("crawler");
var fs = require("fs");

fs.writeFile("./data_crawl_article.txt", "", function(err) {});
fs.writeFile("./data_crawl_skill.txt", "", function(err) {});

var c = new Crawler({
    maxConnections: 10,
    userAgent:
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.139 Safari/537.36",
    // This will be called for each crawled page
    callback: function(error, res, done) {
        if (error) {
            console.log(error);
        } else {
            var $ = res.$;

            $(".list-title.mathjax")
                .text()
                .split("\n")
                .map(value => {
                    if (value.includes(":")) {
                        let data = ""
                        value.split(":").map((value, index) => {
                            if(index > 0){
                                data += value
                            }
                        })
                        fs.appendFile("./data_crawl_article.txt", data + "\n", function(err) {});
                    }
                });

            $(".list-subjects")
                .text()
                .split("\n")
                .map(value => {
                    if (value.includes(":")) {
                        let data = ""
                        value.split(":").map((value, index) => {
                            if(index > 0){
                                data += value
                            }
                        })
                        fs.appendFile("./data_crawl_skill.txt", data + "\n", function(err) {});
                    }
                });
        }
        done();
    }
});

const arr_queue = []

arr_queue.push("https://arxiv.org/list/cs/13?show=2000")

for(let i = 0; i <= 7; i++){
    arr_queue.push("https://arxiv.org/list/cs/14?skip="+(i*2)+"000&show=2000")
}

for(let i = 0; i <= 8; i++){
    arr_queue.push("https://arxiv.org/list/cs/15?skip="+(i*2)+"000&show=2000")
}

for(let i = 0; i <= 10; i++){
    arr_queue.push("https://arxiv.org/list/cs/16?skip="+(i*2)+"000&show=2000")
}

for(let i = 0; i <= 14; i++){
    arr_queue.push("https://arxiv.org/list/cs/17?skip="+(i*2)+"000&show=2000")
}

for(let i = 0; i <= 5; i++){
    arr_queue.push("https://arxiv.org/list/cs/18?skip="+(i*2)+"000&show=2000")
}

c.queue(arr_queue);
