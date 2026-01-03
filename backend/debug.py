# TEMPORARY TEST - Remove after verification
def test_encoding():
    from replay_streamer import comment_time_sec
    
    # Test 47'
    test_comment = {"minute": 47, "extra": 0, "second": 0}
    encoded = comment_time_sec(test_comment)
    
    print("=" * 60)
    print("ENCODING TEST")
    print("=" * 60)
    print(f"Input: 47' +0' 0s")
    print(f"Encoded value: {encoded}")
    print()
    if encoded == 2820:
        print("❌ USING OLD ENCODING: (minute + extra) * 60")
        print("   You need to replace the file!")
    elif encoded == 1470000:
        print("✅ USING NEW ENCODING: half_offset + minute*10000")
        print("   File is correct!")
    else:
        print(f"⚠️  UNKNOWN ENCODING: {encoded}")
    print("=" * 60)

# Run test
test_encoding()