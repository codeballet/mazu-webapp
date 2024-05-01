CREATE TABLE message
(
    message_id INTEGER,
    prompt VARCHAR(255),
    answer VARCHAR(512)
);

CREATE TABLE last
(
    last_message_id INTEGER
);