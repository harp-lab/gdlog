FROM stargazermiao/gdlog-env:11.8

COPY --chown=gdlog:gdlog . /opt/gdlog
WORKDIR /opt/gdlog

# RUN rm -r build
RUN cmake -DCMAKE_BUILD_TYPE:STRING=RelWithDebInfo -Bbuild . && cd build && make -j
# RUN chmod -R 757 /opt/gdlog

# CMD [ "/start.sh" ]
