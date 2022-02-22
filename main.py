from Augmenter import VNIAugmenter,MissingDialectAugmenter, SpellingReplacementAugmenterFinal, SubsituteAugmenter, SpellingReplacementAugmenterBegin, defaultTokenizer
aug = SpellingReplacementAugmenterBegin(tokenizer=defaultTokenizer)
text = r'Theo thông tin rầm rộ, Sơn Tùng - Hải Tú được nghe nói là nge nói là cho là ngồi hạng ghế thương gia của một hãng hàng không. Bên dưới bình luận của bài đăng, rất nhiều bình luận còn xác nhận là đã nhìn thấy cả hai ở sân bay cùng ngày với khẩu trang kín mít...'
augmented_text = aug.augment(text)
print("Original:")
print(text)
print("Augmented Text:")
print(augmented_text)

# aug = SubsituteAugmenter(tokenizer=defaultTokenizer)
# text = r'Tôi đi học.Thích đi học Theo thông tin rầm rộ,tôi Sơn Tùng - Hải Tú được nghe nói là nge nói là cho là ngồi hạng ghế thương gia của một hãng hàng không. Bên dưới bình luận của bài đăng, rất nhiều bình luận còn xác nhận là đã nhìn thấy cả hai ở sân bay cùng ngày với khẩu trang kín mít'
# augmented_text = aug.augment(text)
# print("Original:")
# print(text)
# print("Augmented Text:")
# print(augmented_text)