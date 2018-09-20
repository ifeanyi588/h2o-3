package water.parser;

import water.MemoryManager;

import java.util.Arrays;

/**
 * A schema over a string represented as an array of bytes.
 * This schema enables characters to be skipped inside the string. unlike the basic {@link BufferedString}
 * Skipped characters are not serialized by toString method.
 */
class CharSkippingBufferedString {

    private int[] _skipped;
    private int _skippedWriteIndex;
    private final BufferedString _bufferedString;

    CharSkippingBufferedString() {
        _skipped = new int[0];
        _skippedWriteIndex = 0;
        _bufferedString = new BufferedString();
    }

    protected void addChar() {
        _bufferedString.addChar();
    }

    protected void removeChar() {
        _bufferedString.removeChar();
    }

    protected byte[] getBuffer() {
        return _bufferedString.getBuffer();
    }

    /**
     *
     * @return
     */
    protected boolean isOverflown(){
       return  _bufferedString.getOffset() + _bufferedString.length() > _bufferedString.getBuffer().length;
    }

    protected void addBuff(final byte[] bits) {
        _bufferedString.addBuff(bits);
        _skipped = new int[0];
        _skippedWriteIndex = 0;
    }

    /**
     * Marks a character in the backing array as skipped. Such character is no longer serialized when toString() method
     * is called on this buffer.
     *
     * @param skippedCharIndex Index of the character in the backing array to skip
     */
    protected final void skipIndex(final int skippedCharIndex) {
        _bufferedString.addChar();
        if (_skipped.length == 0 || _skipped[_skipped.length - 1] != -1) {
            _skipped = Arrays.copyOf(_skipped, Math.max(_skipped.length + 1, 1));
        }

        _skipped[_skippedWriteIndex] = skippedCharIndex;
        _skippedWriteIndex++;
    }

    /**
     * A delegate to the underlying {@link StringBuffer}'s set() method.
     *
     * @param buf Buffer to operate with
     * @param off Beginning of the string (offset in the buffer)
     * @param len Length of the string from the offset.
     */
    protected void set(final byte[] buf, final int off, final int len) {
        _skipped = new int[0];
        _skippedWriteIndex = 0;
        _bufferedString.set(buf, off, len);
    }

    /**
     * Converts the current window into byte buffer to a {@link BufferedString}. The resulting new instance of {@link BufferedString}
     * is backed by a newly allocated byte[] buffer sized exactly to fit the desired string represented by current buffer window,
     * excluding the skipped characters.
     *
     * @return An instance of {@link BufferedString} containing only bytes from the original window, without skipped bytes.
     */
    public BufferedString toBufferedString() {
        byte[] buf = MemoryManager.malloc1(_bufferedString.length() - _skipped.length); // Length of the buffer window minus skipped chars

        int nSkipped = 0;
        for (int i = _bufferedString.getOffset(); i < _bufferedString.getOffset() + _bufferedString.length(); i++) {

            if (Arrays.binarySearch(_skipped, i) >= 0) {
                nSkipped++;
                continue;
            }

            final byte character = getBuffer()[i];
            buf[i - _bufferedString.getOffset() - nSkipped] = character;
        }

        return new BufferedString(buf, 0, buf.length);
    }

    /**
     * @return A string representation of the buffer window, excluding skipped characters
     */
    @Override
    public String toString() {
        return toBufferedString().toString();
    }
}
